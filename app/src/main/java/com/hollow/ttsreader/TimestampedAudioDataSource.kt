package com.hollow.ttsreader

import android.net.Uri
import androidx.media3.common.C
import androidx.media3.datasource.BaseDataSource
import androidx.media3.datasource.DataSource
import androidx.media3.datasource.DataSpec
import org.json.JSONException
import org.json.JSONObject
import java.io.IOException
import java.io.InputStream
import java.util.LinkedList
import java.util.Queue
import kotlin.math.min

data class Timestamp(val word: String, val startTime: Double, val endTime: Double)

class TimestampedAudioDataSource(
    private val inputStream: InputStream,
    private val onSessionInit: (String, Int) -> Unit,
    private val onTimestamp: (Timestamp) -> Unit,
    private val onAudioChunk: (Int) -> Unit,
    private val onEndOfStream: () -> Unit
) : BaseDataSource(true) {

    private val boundary = "--CHUNK_BOUNDARY--".toByteArray()
    private var uri: Uri? = null
    private var opened = false

    private val audioBuffer: Queue<Byte> = LinkedList()
    private var streamEnded = false

    override fun open(dataSpec: DataSpec): Long {
        uri = dataSpec.uri
        transferInitializing(dataSpec)
        opened = true
        transferStarted(dataSpec)
        return C.LENGTH_UNSET.toLong()
    }

    override fun read(target: ByteArray, offset: Int, length: Int): Int {
        if (length == 0) return 0
        if (streamEnded && audioBuffer.isEmpty()) return C.RESULT_END_OF_INPUT

        while (audioBuffer.size < length && !streamEnded) {
            readNextPart()
        }

        val bytesToRead = min(length, audioBuffer.size)
        if (bytesToRead == 0) {
            return if (streamEnded) C.RESULT_END_OF_INPUT else 0
        }

        for (i in 0 until bytesToRead) {
            target[offset + i] = audioBuffer.poll()
        }
        return bytesToRead
    }

    private fun readNextPart() {
        val partData = readUntilBoundary()
        if (partData == null || partData.isEmpty()) {
            if (!streamEnded) {
                streamEnded = true
                onEndOfStream()
            }
            return
        }

        try {
            val jsonString = String(partData, Charsets.UTF_8)
            val json = JSONObject(jsonString)

            when (json.optString("type")) {
                "init" -> {
                    val sessionId = json.getString("session_id")
                    val totalChunks = json.getInt("total_chunks")
                    onSessionInit(sessionId, totalChunks)
                }
                "timestamp" -> {
                    val word = json.getString("word")
                    val startTime = json.getDouble("start_time")
                    val endTime = json.getDouble("end_time")
                    onTimestamp(Timestamp(word, startTime, endTime))
                }
                "audio" -> {
                    val chunkIndex = json.getInt("chunk_index")
                    onAudioChunk(chunkIndex)
                    // The audio data follows the JSON part, so we read it directly.
                    val audioData = readUntilBoundary()
                    audioData?.forEach { audioBuffer.offer(it) }
                }
                "end_of_stream" -> {
                    streamEnded = true
                    onEndOfStream()
                }
                else -> {
                    // Unknown JSON part, ignore.
                }
            }
        } catch (e: JSONException) {
            // This should not happen if the stream is well-formed.
            // If it does, we might be out of sync. For now, we assume it's audio.
            partData.forEach { audioBuffer.offer(it) }
        }
    }

    private fun readUntilBoundary(): ByteArray? {
        val buffer = mutableListOf<Byte>()
        val window = LinkedList<Byte>()

        while (true) {
            val byteInt = try {
                inputStream.read()
            } catch (e: IOException) {
                return null // Stream closed or error
            }

            if (byteInt == -1) {
                return if (buffer.isNotEmpty()) buffer.toByteArray() else null // End of stream
            }

            val byte = byteInt.toByte()
            buffer.add(byte)

            if (window.size == boundary.size) {
                window.removeFirst()
            }
            window.addLast(byte)

            if (window.size == boundary.size) {
                var match = true
                for(i in 0 until boundary.size) {
                    if (window[i] != boundary[i]) {
                        match = false
                        break
                    }
                }
                if (match) {
                    return buffer.subList(0, buffer.size - boundary.size).toByteArray()
                }
            }
        }
    }

    override fun getUri(): Uri? = uri

    override fun close() {
        if (opened) {
            opened = false
            transferEnded()
            try {
                inputStream.close()
            } catch (e: IOException) {
                // ignore
            }
        }
    }

    class Factory(
        private val inputStream: InputStream,
        private val onSessionInit: (String, Int) -> Unit,
        private val onTimestamp: (Timestamp) -> Unit,
        private val onAudioChunk: (Int) -> Unit,
        private val onEndOfStream: () -> Unit
    ) : DataSource.Factory {
        override fun createDataSource(): DataSource {
            return TimestampedAudioDataSource(inputStream, onSessionInit, onTimestamp, onAudioChunk, onEndOfStream)
        }
    }
}