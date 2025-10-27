package com.hollow.ttsreader

import android.content.Context
import com.hollow.ttsreader.Book
import com.hollow.ttsreader.BookManager
import com.hollow.ttsreader.WordTimestamp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okhttp3.logging.HttpLoggingInterceptor
import java.io.File
import java.io.FileOutputStream
import java.util.UUID
import java.util.zip.ZipInputStream


// ==========================================
// SERVER API - Communicates with Colab
// ==========================================
object ServerAPI {

    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }

    private val client = OkHttpClient.Builder()
        .addInterceptor(loggingInterceptor)
        .connectTimeout(5, java.util.concurrent.TimeUnit.MINUTES)
        .readTimeout(10, java.util.concurrent.TimeUnit.MINUTES)
        .writeTimeout(5, java.util.concurrent.TimeUnit.MINUTES)
        .build()

    private val json = Json { ignoreUnknownKeys = true }

    suspend fun convertBook(
        title: String,
        text: String,
        context: Context,
        serverUrl: String, // <-- ADDED PARAMETER
        onProgress: (String) -> Unit
    ): Book = withContext(Dispatchers.IO) {

        // Create JSON payload
        val jsonPayload = buildJsonObject {
            put("title", title)
            put("text", text)
        }.toString()

        val requestBody = jsonPayload.toRequestBody("application/json".toMediaType())

        val request = Request.Builder()
            .url("$serverUrl/convert") // <-- USE THE PARAMETER
            .post(requestBody)
            .build()

        onProgress("Uploading to server...")

        val response = client.newCall(request).execute()

        if (!response.isSuccessful) {
            val errorBody = response.body?.string() ?: "Unknown error"
            throw Exception("Server error ${response.code}: $errorBody")
        }

        onProgress("Downloading converted files...")

        // Save the ZIP file
        val bookId = UUID.randomUUID().toString()
        val bookManager = BookManager(context)
        val bookDir = bookManager.createBookDirectory(bookId)

        val zipFile = File(bookDir, "download.zip")
        response.body?.byteStream()?.use { input ->
            FileOutputStream(zipFile).use { output ->
                input.copyTo(output)
            }
        }

        onProgress("Extracting files...")

        // Unzip the files
        var audioPath = ""
        var timestampsPath = ""
        var textPath = ""

        ZipInputStream(zipFile.inputStream()).use { zipInput ->
            var entry = zipInput.nextEntry
            while (entry != null) {
                val file = File(bookDir, entry.name)

                if (!entry.isDirectory) {
                    FileOutputStream(file).use { output ->
                        zipInput.copyTo(output)
                    }

                    when (entry.name) {
                        "final_audio.mp3" -> audioPath = file.absolutePath
                        "timestamps.json" -> timestampsPath = file.absolutePath
                        "book_text.txt" -> textPath = file.absolutePath
                    }
                }

                zipInput.closeEntry()
                entry = zipInput.nextEntry
            }
        }

        // Delete zip file
        zipFile.delete()

        if (audioPath.isEmpty() || timestampsPath.isEmpty() || textPath.isEmpty()) {
            throw Exception("Conversion failed: Zip file did not contain all required files.")
        }

        onProgress("Finalizing...")

        // Calculate metadata
        val bookText = File(textPath).readText()
        val wordCount = bookText.split(Regex("\\s+")).size

        // Get audio duration from timestamps
        val timestamps = json.decodeFromString<List<WordTimestamp>>(
            File(timestampsPath).readText()
        )
        val durationSeconds = timestamps.lastOrNull()?.end?.toInt() ?: 0
        val durationFormatted = formatDuration(durationSeconds)

        // Create Book object
        Book(
            id = bookId,
            title = title,
            audioPath = audioPath,
            timestampsPath = timestampsPath,
            textPath = textPath,
            wordCount = wordCount,
            duration = durationFormatted
        )
    }

    suspend fun streamAudio(
        text: String,
        serverUrl: String
    ): Response = withContext(Dispatchers.IO) {

        val jsonPayload = buildJsonObject {
            put("text", text)
        }.toString()

        val requestBody = jsonPayload.toRequestBody("application/json".toMediaType())

        val request = Request.Builder()
            .url("$serverUrl/stream")
            .post(requestBody)
            .build()

        // Retry loop for transient DNS/network issues
        val maxRetries = 3
        var attempt = 0
        var lastException: Exception? = null

        while (attempt < maxRetries) {
            try {
                return@withContext client.newCall(request).execute()
            } catch (e: Exception) {
                lastException = e
                // If it's a DNS / UnknownHost, wait and retry a couple times before failing
                if (e is java.net.UnknownHostException || e is java.net.SocketTimeoutException || e is java.io.IOException) {
                    attempt++
                    if (attempt >= maxRetries) break
                    try {
                        Thread.sleep((1000L * attempt))
                    } catch (ie: InterruptedException) {
                        // restore interrupt
                        Thread.currentThread().interrupt()
                        break
                    }
                    continue
                } else {
                    // Non-network exception, rethrow
                    throw e
                }
            }
        }

        // If we reached here, retries exhausted
        throw lastException ?: Exception("Unknown network error")
    }

    suspend fun sendChunkReceivedConfirmation(serverUrl: String, sessionId: String, chunkIndex: Int) = withContext(Dispatchers.IO) {
        try {
            val jsonPayload = buildJsonObject {
                put("session_id", sessionId)
                put("chunk_index", chunkIndex)
            }.toString()
            val requestBody = jsonPayload.toRequestBody("application/json".toMediaType())
            val request = Request.Builder()
                .url("$serverUrl/chunk_received")
                .post(requestBody)
                .build()
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    // Log error or handle unsuccessful confirmation
                }
            }
        } catch (e: Exception) {
            // Handle network exception
        }
    }

    private fun formatDuration(seconds: Int): String {
        val hours = seconds / 3600
        val minutes = (seconds % 3600) / 60
        val secs = seconds % 60

        return if (hours > 0) {
            String.format("%d:%02d:%02d", hours, minutes, secs)
        } else {
            String.format("%02d:%02d", minutes, secs)
        }
    }

    suspend fun checkServerStatus(serverUrl: String): Boolean = withContext(Dispatchers.IO) { // <-- ADDED PARAMETER
        if (serverUrl.isBlank()) return@withContext false
        try {
            val request = Request.Builder()
                .url(serverUrl) // <-- USE THE PARAMETER
                .get()
                .build()

            val response = client.newCall(request).execute()
            response.isSuccessful
        } catch (e: Exception) {
            false
        }
    }
}
