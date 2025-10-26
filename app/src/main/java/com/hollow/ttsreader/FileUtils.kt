package com.hollow.ttsreader

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import java.lang.Exception // Or more specific exceptions
import java.lang.IllegalArgumentException

// ==========================================
// FILE UTILITIES - Extract text from files
// ==========================================
object FileUtils {

    fun getFileName(context: Context, uri: Uri): String? {
        var name: String? = null
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (nameIndex != -1) {
                    name = cursor.getString(nameIndex)
                }
            }
        }
        return name
    }

    fun extractText(context: Context, uri: Uri): String {
        val fileName = getFileName(context, uri) ?: ""

        return try {
            when {
                // PDF support removed
                // fileName.endsWith(".pdf", ignoreCase = true) -> extractFromPdf(context, uri)

                // EPUB support removed
                // fileName.endsWith(".epub", ignoreCase = true) -> extractFromEpub(context, uri)

                fileName.endsWith(".txt", ignoreCase = true) -> extractFromTxt(context, uri)
                else -> throw IllegalArgumentException("Unsupported file format. Please use TXT only.")
            }
        } catch (e: Exception) {
            e.printStackTrace()
            throw Exception("Failed to extract text from file: ${e.message}")
        }
    }

    // PDF extraction function - commented out
    /*
    private fun extractFromPdf(context: Context, uri: Uri): String {
        context.contentResolver.openInputStream(uri)?.use { input ->
            val tempFile = JavaFile.createTempFile("upload", ".pdf", context.cacheDir) // Use JavaFile
            tempFile.outputStream().use { output ->
                input.copyTo(output)
            }
            // Assuming PDDocument can load from a JavaFile object
            val document = PDDocument.load(tempFile) // Use tempFile (which is a JavaFile)
            val stripper = PDFTextStripper()
            val text = stripper.getText(document)
            document.close()
            tempFile.delete()
            return text
        } ?: throw Exception("Failed to open PDF file")
    }
    */

    // EPUB extraction function - commented out
    /*
    private fun extractFromEpub(context: Context, uri: Uri): String {
        // ... (code remains commented out) ...
    }
    */

    private fun extractFromTxt(context: Context, uri: Uri): String {
        context.contentResolver.openInputStream(uri)?.use { input ->
            return input.bufferedReader(Charsets.UTF_8).use { it.readText() }
        } ?: throw Exception("Failed to open TXT file")
    }
}

// ==========================================
// TEXT CLEANER - Clean extracted text
// ==========================================
object TextCleaner {

    fun clean(rawText: String): String {
        var cleanedText = rawText

        // Normalize line endings and multiple spaces
        cleanedText = cleanedText.replace(Regex("\r\n|\r"), "\n")
        cleanedText = cleanedText.replace(Regex("[ \t]+"), " ")

        // Remove lines with just numbers (common page numbers)
        cleanedText = cleanedText.lines()
            .filterNot { it.trim().matches(Regex("^\\d+$")) }
            .joinToString("\n")

        // Remove excessive blank lines
        cleanedText = cleanedText.replace(Regex("\n{3,}"), "\n\n")

        return cleanedText.trim()
    }
}