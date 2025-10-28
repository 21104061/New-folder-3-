package com.hollow.ttsreader

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.work.CoroutineWorker
import androidx.work.ForegroundInfo
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.io.FileOutputStream
import java.util.zip.ZipInputStream

class ConversionWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    private val notificationManager =
        context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

    companion object {
        const val KEY_BOOK_ID = "book_id"
        const val KEY_BOOK_TITLE = "book_title"
        const val KEY_SERVER_URL = "server_url"
        const val KEY_WORD_COUNT = "word_count"

        const val CHANNEL_ID = "book_conversion_channel"
        const val NOTIFICATION_ID = 1001

        private const val TAG = "ConversionWorker"
    }

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val bookId = inputData.getString(KEY_BOOK_ID) ?: run {
            Log.e(TAG, "Missing book ID")
            return@withContext Result.failure(workDataOf("error" to "Missing book ID"))
        }

        val title = inputData.getString(KEY_BOOK_TITLE) ?: run {
            Log.e(TAG, "Missing book title")
            return@withContext Result.failure(workDataOf("error" to "Missing title"))
        }

        // Read text from the saved file instead of input data (to avoid 10KB WorkManager limit)
        val bookManager = BookManager(applicationContext)
        val book = bookManager.getBook(bookId) ?: run {
            Log.e(TAG, "Book not found: $bookId")
            return@withContext Result.failure(workDataOf("error" to "Book not found"))
        }
        
        val text = try {
            File(book.textPath).readText()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to read book text: ${e.message}")
            return@withContext Result.failure(workDataOf("error" to "Failed to read book text"))
        }

        val serverUrl = inputData.getString(KEY_SERVER_URL) ?: run {
            Log.e(TAG, "Missing server URL")
            return@withContext Result.failure(workDataOf("error" to "Missing server URL"))
        }

        val wordCount = inputData.getInt(KEY_WORD_COUNT, 0)

        try {
            createNotificationChannel()
            setForeground(createForegroundInfo(title, "Starting conversion..."))

            Log.d(TAG, "Starting conversion for book: $title (ID: $bookId)")

            // Update status to CONVERTING
            bookManager.updateBookStatus(bookId, BookStatus.CONVERTING)

            // Start conversion
            setForeground(createForegroundInfo(title, "Sending to server..."))

            Log.d(TAG, "Starting async conversion with serverUrl: $serverUrl")

            // Start async conversion
            val jobResponse = ServerAPI.startAsyncConversion(
                title = title,
                text = text,
                serverUrl = serverUrl
            )

            Log.d(TAG, "Async job started: ${jobResponse.job_id}")

            // Update book with job ID
            val currentBooks = bookManager.getBooks().toMutableList()
            val bookIndex = currentBooks.indexOfFirst { it.id == bookId }
            if (bookIndex != -1) {
                currentBooks[bookIndex] = currentBooks[bookIndex].copy(jobId = jobResponse.job_id)
                val metadataFile = File(applicationContext.filesDir, "books_metadata.json")
                val jsonString = Json.encodeToString(currentBooks)
                metadataFile.writeText(jsonString)
            }

            // Poll for completion
            setForeground(createForegroundInfo(title, "Waiting for server processing..."))
            
            var attempts = 0
            val maxAttempts = 360 // 3 hours with 30-second intervals
            
            // Get initial status
            var jobStatus = try {
                ServerAPI.getJobStatus(jobResponse.job_id, serverUrl)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get initial job status: ${e.message}")
                bookManager.updateBookStatus(bookId, BookStatus.ERROR)
                showErrorNotification(title, "Failed to start conversion")
                return@withContext Result.failure(workDataOf("error" to "Failed to start conversion"))
            }
            
            while (jobStatus.status in listOf("queued", "processing") && attempts < maxAttempts) {
                delay(30000) // Wait 30 seconds between polls
                attempts++
                
                try {
                    jobStatus = ServerAPI.getJobStatus(jobResponse.job_id, serverUrl)
                    Log.d(TAG, "Job status: ${jobStatus.status}, progress: ${jobStatus.progress}")
                    
                    // Update notification with progress
                    val progressText = jobStatus.progress ?: jobStatus.status
                    setForeground(createForegroundInfo(title, progressText))
                    
                    when (jobStatus.status) {
                        "completed" -> break
                        "failed" -> {
                            val error = jobStatus.error ?: "Unknown error"
                            Log.e(TAG, "Job failed: $error")
                            bookManager.updateBookStatus(bookId, BookStatus.ERROR)
                            showErrorNotification(title, "Conversion failed: $error")
                            return@withContext Result.failure(workDataOf("error" to error))
                        }
                    }
                    
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to get job status (attempt $attempts): ${e.message}")
                    if (attempts >= maxAttempts) {
                        Log.e(TAG, "Max polling attempts reached")
                        bookManager.updateBookStatus(bookId, BookStatus.ERROR)
                        showErrorNotification(title, "Conversion timeout after 3 hours")
                        return@withContext Result.failure(workDataOf("error" to "Timeout"))
                    }
                }
                
            }

            if (attempts >= maxAttempts) {
                Log.e(TAG, "Conversion timed out")
                bookManager.updateBookStatus(bookId, BookStatus.ERROR)
                showErrorNotification(title, "Conversion timeout")
                return@withContext Result.failure(workDataOf("error" to "Timeout"))
            }

            // Download completed file
            setForeground(createForegroundInfo(title, "Downloading completed file..."))
            
            val response = ServerAPI.downloadCompletedJob(jobResponse.job_id, serverUrl)
            
            Log.d(TAG, "Download response code: ${response.code}")

            if (!response.isSuccessful) {
                val errorBody = response.body?.string() ?: "Unknown error (empty response)"
                Log.e(TAG, "Download failed: ${response.code} - $errorBody")

                bookManager.updateBookStatus(bookId, BookStatus.ERROR)
                showErrorNotification(title, "Download failed: ${response.code} - $errorBody")

                return@withContext Result.failure(
                    workDataOf("error" to "Download failed: ${response.code} - $errorBody")
                )
            }

            // Check if response has a body
            val responseBody = response.body
            if (responseBody == null) {
                Log.e(TAG, "Response body is null")
                bookManager.updateBookStatus(bookId, BookStatus.ERROR)
                showErrorNotification(title, "Empty response from server")
                return@withContext Result.failure(
                    workDataOf("error" to "Empty response from server")
                )
            }

            // Update status to DOWNLOADING
            bookManager.updateBookStatus(bookId, BookStatus.DOWNLOADING)
            setForeground(createForegroundInfo(title, "Downloading converted file..."))

            // Save the ZIP file
            val bookDir = bookManager.createBookDirectory(bookId)
            val zipFile = File(bookDir, "download.zip")

            Log.d(TAG, "Saving ZIP to: ${zipFile.absolutePath}")

            var bytesDownloaded = 0L
            responseBody.byteStream().use { input ->
                FileOutputStream(zipFile).use { output ->
                    val buffer = ByteArray(8192)
                    var bytes = input.read(buffer)
                    while (bytes >= 0) {
                        output.write(buffer, 0, bytes)
                        bytesDownloaded += bytes

                        // Update notification every 100KB
                        if (bytesDownloaded % 102400 == 0L) {
                            setForeground(createForegroundInfo(
                                title,
                                "Downloaded ${bytesDownloaded / 1024}KB..."
                            ))
                        }

                        bytes = input.read(buffer)
                    }
                }
            }

            Log.d(TAG, "Download complete: $bytesDownloaded bytes")

            if (!zipFile.exists() || zipFile.length() == 0L) {
                Log.e(TAG, "ZIP file is empty or doesn't exist")
                bookManager.updateBookStatus(bookId, BookStatus.ERROR)
                showErrorNotification(title, "Downloaded file is empty")
                return@withContext Result.failure(
                    workDataOf("error" to "Downloaded file is empty")
                )
            }

            // Extract files
            setForeground(createForegroundInfo(title, "Extracting files..."))
            Log.d(TAG, "Extracting ZIP file...")

            var audioPath = ""
            var timestampsPath = ""
            var textPath = ""
            var speakersMetadataPath = ""

            try {
                ZipInputStream(zipFile.inputStream()).use { zipInput ->
                    var entry = zipInput.nextEntry
                    var entryCount = 0

                    while (entry != null) {
                        entryCount++
                        val file = File(bookDir, entry.name)
                        Log.d(TAG, "Extracting: ${entry.name} (${entry.size} bytes)")

                        if (!entry.isDirectory) {
                            FileOutputStream(file).use { output ->
                                zipInput.copyTo(output)
                            }

                            when (entry.name) {
                                "final_audio.mp3" -> audioPath = file.absolutePath
                                "timestamps.json" -> timestampsPath = file.absolutePath
                                "book_text.txt" -> textPath = file.absolutePath
                                "speakers_metadata.json" -> speakersMetadataPath = file.absolutePath
                            }
                        }

                        zipInput.closeEntry()
                        entry = zipInput.nextEntry
                    }

                    Log.d(TAG, "Extracted $entryCount entries from ZIP")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error extracting ZIP: ${e.message}", e)
                bookManager.updateBookStatus(bookId, BookStatus.ERROR)
                showErrorNotification(title, "Failed to extract files: ${e.message}")
                return@withContext Result.failure(
                    workDataOf("error" to "Extraction failed: ${e.message}")
                )
            }

            // Delete zip file
            zipFile.delete()
            Log.d(TAG, "Deleted ZIP file")

            if (audioPath.isEmpty() || textPath.isEmpty()) {
                Log.e(TAG, "Missing required files - audio: $audioPath, text: $textPath")
                bookManager.updateBookStatus(bookId, BookStatus.ERROR)
                showErrorNotification(title, "Downloaded file is incomplete (missing audio or text)")
                return@withContext Result.failure(
                    workDataOf("error" to "Incomplete download - missing required files")
                )
            }

            // Get audio duration
            val timestamps = if (File(timestampsPath).exists()) {
                try {
                    val json = File(timestampsPath).readText()
                    Json.decodeFromString<List<WordTimestamp>>(json)
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to parse timestamps: ${e.message}")
                    emptyList()
                }
            } else {
                Log.w(TAG, "No timestamps file found")
                emptyList()
            }

            val durationSeconds = timestamps.lastOrNull()?.end?.toInt() ?: 0
            val durationFormatted = formatDuration(durationSeconds)

            Log.d(TAG, "Audio duration: $durationFormatted ($durationSeconds seconds)")

            // Create final Book object with READY status
            val finalBook = Book(
                id = bookId,
                title = title,
                audioPath = audioPath,
                timestampsPath = timestampsPath,
                textPath = textPath,
                wordCount = wordCount,
                duration = durationFormatted,
                status = BookStatus.READY
            )

            // Update book in metadata
            val updatedBooks = bookManager.getBooks().toMutableList() // Fixing duplicate currentBooks declaration
            val index = updatedBooks.indexOfFirst { it.id == bookId }
            if (index != -1) {
                updatedBooks[index] = finalBook
                val metadataFile = File(applicationContext.filesDir, "books_metadata.json")
                val jsonString = Json.encodeToString(updatedBooks)
                metadataFile.writeText(jsonString)
                Log.d(TAG, "Updated book metadata")
            } else {
                Log.w(TAG, "Book not found in metadata, index: $index")
            }

            // Show success notification
            showSuccessNotification(title)
            Log.d(TAG, "Conversion completed successfully")

            Result.success()

        } catch (e: Exception) {
            Log.e(TAG, "Conversion failed with exception", e)
            e.printStackTrace()
            bookManager.updateBookStatus(bookId, BookStatus.ERROR)
            showErrorNotification(title, e.message ?: "Unknown error occurred")
            Result.failure(
                workDataOf("error" to (e.message ?: "Unknown error"))
            )
        }
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Book Conversion",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Shows progress of book conversion"
            }
            notificationManager.createNotificationChannel(channel)
        }
    }

    private fun createForegroundInfo(title: String, progress: String): ForegroundInfo {
        val notification = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
            .setContentTitle("Converting: $title")
            .setContentText(progress)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .setProgress(0, 0, true)
            .build()

        return ForegroundInfo(NOTIFICATION_ID, notification)
    }

    private fun showSuccessNotification(title: String) {
        val intent = Intent(applicationContext, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        }

        val pendingIntent = PendingIntent.getActivity(
            applicationContext,
            0,
            intent,
            PendingIntent.FLAG_IMMUTABLE
        )

        val notification = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
            .setContentTitle("✅ Conversion Complete")
            .setContentText("$title is ready to play!")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setAutoCancel(true)
            .setContentIntent(pendingIntent)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .build()

        notificationManager.notify(NOTIFICATION_ID + 1, notification)
    }

    private fun showErrorNotification(title: String, error: String) {
        val notification = NotificationCompat.Builder(applicationContext, CHANNEL_ID)
            .setContentTitle("❌ Conversion Failed")
            .setContentText("$title: $error")
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setStyle(NotificationCompat.BigTextStyle().bigText("$title: $error"))
            .build()

        notificationManager.notify(NOTIFICATION_ID + 2, notification)
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
}