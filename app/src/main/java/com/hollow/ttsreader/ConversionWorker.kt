package com.hollow.ttsreader

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.work.CoroutineWorker
import androidx.work.ForegroundInfo
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import kotlinx.coroutines.Dispatchers
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
        const val KEY_BOOK_TEXT = "book_text"
        const val KEY_SERVER_URL = "server_url"
        const val KEY_WORD_COUNT = "word_count"
        
        const val CHANNEL_ID = "book_conversion_channel"
        const val NOTIFICATION_ID = 1001
    }

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val bookId = inputData.getString(KEY_BOOK_ID) ?: return@withContext Result.failure()
        val title = inputData.getString(KEY_BOOK_TITLE) ?: return@withContext Result.failure()
        val text = inputData.getString(KEY_BOOK_TEXT) ?: return@withContext Result.failure()
        val serverUrl = inputData.getString(KEY_SERVER_URL) ?: return@withContext Result.failure()
        val wordCount = inputData.getInt(KEY_WORD_COUNT, 0)

        val bookManager = BookManager(applicationContext)

        try {
            createNotificationChannel()
            setForeground(createForegroundInfo(title, "Starting conversion..."))

            // Update status to CONVERTING
            bookManager.updateBookStatus(bookId, BookStatus.CONVERTING)

            // Start conversion
            setForeground(createForegroundInfo(title, "Converting to audio..."))

            val response = ServerAPI.convertBookRaw(
                title = title,
                text = text,
                serverUrl = serverUrl
            )

            if (!response.isSuccessful) {
                val errorBody = response.body?.string() ?: "Unknown error"
                bookManager.updateBookStatus(bookId, BookStatus.ERROR)
                
                showErrorNotification(title, "Conversion failed: $errorBody")
                return@withContext Result.failure(
                    workDataOf("error" to errorBody)
                )
            }

            // Update status to DOWNLOADING
            bookManager.updateBookStatus(bookId, BookStatus.DOWNLOADING)
            setForeground(createForegroundInfo(title, "Downloading..."))

            // Save the ZIP file
            val bookDir = bookManager.createBookDirectory(bookId)
            val zipFile = File(bookDir, "download.zip")
            
            response.body?.byteStream()?.use { input ->
                FileOutputStream(zipFile).use { output ->
                    input.copyTo(output)
                }
            }

            // Extract files
            setForeground(createForegroundInfo(title, "Extracting files..."))

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

            if (audioPath.isEmpty() || textPath.isEmpty()) {
                bookManager.updateBookStatus(bookId, BookStatus.ERROR)
                showErrorNotification(title, "Downloaded file is incomplete")
                return@withContext Result.failure(
                    workDataOf("error" to "Incomplete download")
                )
            }

            // Get audio duration
            val timestamps = if (File(timestampsPath).exists()) {
                val json = File(timestampsPath).readText()
                Json.decodeFromString<List<WordTimestamp>>(json)
            } else {
                emptyList()
            }
            
            val durationSeconds = timestamps.lastOrNull()?.end?.toInt() ?: 0
            val durationFormatted = formatDuration(durationSeconds)

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
            val currentBooks = bookManager.getBooks().toMutableList()
            val index = currentBooks.indexOfFirst { it.id == bookId }
            if (index != -1) {
                currentBooks[index] = finalBook
                val metadataFile = File(applicationContext.filesDir, "books_metadata.json")
                val jsonString = Json.encodeToString(currentBooks)
                metadataFile.writeText(jsonString)
            }

            // Show success notification
            showSuccessNotification(title)

            Result.success()

        } catch (e: Exception) {
            e.printStackTrace()
            bookManager.updateBookStatus(bookId, BookStatus.ERROR)
            showErrorNotification(title, e.message ?: "Unknown error")
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