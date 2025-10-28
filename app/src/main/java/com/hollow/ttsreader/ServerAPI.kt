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
        .retryOnConnectionFailure(true)
        .connectTimeout(2, java.util.concurrent.TimeUnit.MINUTES)
        .readTimeout(60, java.util.concurrent.TimeUnit.MINUTES)
        .writeTimeout(60, java.util.concurrent.TimeUnit.MINUTES)
        .build()

    private val json = Json { ignoreUnknownKeys = true }

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

    suspend fun convertBookRaw(
        title: String,
        text: String,
        serverUrl: String
    ): Response = withContext(Dispatchers.IO) {
        val jsonPayload = buildJsonObject {
            put("text", text)
            put("title", title)
        }.toString()

        val requestBody = jsonPayload.toRequestBody("application/json".toMediaType())

        val request = Request.Builder()
            .url("$serverUrl/convert")
            .post(requestBody)
            .build()

        return@withContext client.newCall(request).execute()
    }
}
