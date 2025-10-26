package com.hollow.ttsreader

import android.content.Context
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File

// Data class for a single book
@Serializable
data class Book(
    val id: String,
    val title: String,
    val audioPath: String,
    val timestampsPath: String,
    val textPath: String,
    val wordCount: Int,
    val duration: String,
    val dateAdded: Long = System.currentTimeMillis()
) {
    fun getText(): String {
        return File(textPath).readText()
    }

    fun getTimestamps(): List<WordTimestamp> {
        val json = File(timestampsPath).readText()
        return Json.decodeFromString<List<WordTimestamp>>(json)
    }
}

// Data class for a single word with its timestamp
@Serializable
data class WordTimestamp(
    val word: String,
    val start: Double,  // seconds
    val end: Double     // seconds
)

// Sealed class to represent the different screens in the app
sealed class Screen {
    object Library : Screen()
    object Upload : Screen()
    object Settings : Screen()
    data class Player(val book: Book) : Screen()
    data class StreamPlayer(val text: String) : Screen()
}

class BookManager(private val context: Context) {
    private val json = Json {
        prettyPrint = true
        ignoreUnknownKeys = true
    }

    private val booksDir = File(context.filesDir, "books").apply { mkdirs() }
    private val metadataFile = File(context.filesDir, "books_metadata.json")

    fun getBooks(): List<Book> {
        if (!metadataFile.exists()) return emptyList()

        return try {
            val jsonString = metadataFile.readText()
            Json.decodeFromString<List<Book>>(jsonString)
        } catch (e: Exception) {
            e.printStackTrace()
            emptyList()
        }
    }

    fun saveBook(book: Book) {
        val currentBooks = getBooks().toMutableList()
        currentBooks.add(book)

        val jsonString = Json.encodeToString(currentBooks)
        metadataFile.writeText(jsonString)
    }

    fun deleteBook(bookId: String) {
        val currentBooks = getBooks().toMutableList()
        val book = currentBooks.find { it.id == bookId } ?: return

        File(book.audioPath).delete()
        File(book.timestampsPath).delete()
        File(book.textPath).delete()
        File(booksDir, bookId).deleteRecursively()

        currentBooks.removeIf { it.id == bookId }
        val jsonString = Json.encodeToString(currentBooks)
        metadataFile.writeText(jsonString)
    }

    fun createBookDirectory(bookId: String): File {
        return File(booksDir, bookId).apply { mkdirs() }
    }
}
