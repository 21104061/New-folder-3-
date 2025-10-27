package com.hollow.ttsreader

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.*
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.draw.rotate
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.media3.common.C
import androidx.media3.common.Format
import androidx.media3.common.MediaItem
import androidx.media3.common.MediaMetadata
import androidx.media3.common.MimeTypes
import androidx.media3.common.PlaybackParameters
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.common.TrackSelectionParameters
import androidx.media3.datasource.DataSource
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.source.ProgressiveMediaSource
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONException
import org.json.JSONObject
import java.io.IOException
import java.io.InputStream
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import java.util.LinkedList
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.roundToInt

// Helper function to read from the stream until a boundary is found or timeout occurs
private fun readUntilBoundary(inputStream: InputStream, boundary: ByteArray): ByteArray? {
    val buffer = mutableListOf<Byte>()
    val window = LinkedList<Byte>()
    val startTime = System.currentTimeMillis()
    val timeout = 30_000 // 30 seconds timeout

    while (System.currentTimeMillis() - startTime < timeout) {
        // Check for server connection issues
        if (inputStream.available() == 0) {
            Thread.sleep(100) // Give server some time to send more data
            if (inputStream.available() == 0) {
                throw IOException("No data available from server")
            }
        }

        val byteInt = try {
            inputStream.read()
        } catch (e: IOException) {
            return null
        }

        if (byteInt == -1) {
            return if (buffer.isNotEmpty()) buffer.toByteArray() else null
        }

        val byte = byteInt.toByte()
        buffer.add(byte)

        if (window.size == boundary.size) {
            window.removeFirst()
        }
        window.addLast(byte)

        if (window.size == boundary.size) {
            var match = true
            for (i in 0 until boundary.size) {
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
    
    // If we reach here, timeout occurred
    throw IOException("Timeout waiting for boundary")
}

// Read exactly `size` bytes from the input stream. Returns null on premature EOF or error.
private fun readExactly(inputStream: InputStream, size: Int): ByteArray? {
    if (size <= 0) return null
    val buffer = ByteArray(size)
    var offset = 0
    var retryCount = 0
    val maxRetries = 3
    val retryDelay = 100L // milliseconds

    while (offset < size && retryCount < maxRetries) {
        try {
            // Check for server connection issues
            if (inputStream.available() == 0) {
                Thread.sleep(retryDelay)
                if (inputStream.available() == 0) {
                    retryCount++
                    continue
                }
            }

            val read = inputStream.read(buffer, offset, size - offset)
            if (read == -1) {
                // EOF before reading all bytes
                return if (offset > 0) buffer.copyOf(offset) else null
            }
            offset += read
            retryCount = 0 // Reset retry count on successful read

        } catch (e: IOException) {
            retryCount++
            Thread.sleep(retryDelay)
        }
    }

    return if (offset == size) buffer else null
}

// Custom DataSource that reads from a buffer queue
@androidx.annotation.OptIn(androidx.media3.common.util.UnstableApi::class)
class BufferedAudioDataSource(
    private val audioBuffer: LinkedBlockingQueue<ByteArray>,
    private val format: Format,
    private val onBufferEmpty: () -> Boolean // Returns true if stream is complete
) : androidx.media3.datasource.BaseDataSource(true) {

    private var opened = false
    private var currentChunk: ByteArray? = null
    private var currentPosition = 0
    private var bytesRemaining = C.LENGTH_UNSET.toLong()
    
    init {
        android.util.Log.d("BufferedAudioDataSource", "Initializing with format: ${format.sampleRate}Hz, ${format.channelCount} channels")
    }

    override fun open(dataSpec: androidx.media3.datasource.DataSpec): Long {
        android.util.Log.d("BufferedAudioDataSource", "Opening data source with format: ${format.sampleRate}Hz, ${format.channelCount} channels")
        opened = true
        transferInitializing(dataSpec)
        transferStarted(dataSpec)
        return androidx.media3.common.C.LENGTH_UNSET.toLong()
    }

    override fun read(target: ByteArray, offset: Int, length: Int): Int {
        if (!opened) throw IOException("Attempt to read from closed DataSource")
        if (length == 0) return 0
        
        android.util.Log.d("BufferedAudioDataSource", "Read request: offset=$offset length=$length")

        // Work with a local reference to avoid smart-cast problems
        var chunk = currentChunk

        // If current chunk is exhausted or null, get a new one.
        if (chunk == null || currentPosition >= chunk.size) {
            // Quick check: if producer has completed and queue empty, return EOF
            if (onBufferEmpty() && audioBuffer.isEmpty()) {
                return androidx.media3.common.C.RESULT_END_OF_INPUT
            }

            // Block until next chunk is available or interrupted. Use take() so ExoPlayer
            // will wait for the server to deliver data instead of failing fast.
            try {
                val newChunk = audioBuffer.take()
                // If a zero-length sentinel was offered to signal completion, treat as EOF
                if (newChunk == null || newChunk.isEmpty()) {
                    return androidx.media3.common.C.RESULT_END_OF_INPUT
                }
                currentChunk = newChunk
                chunk = newChunk
            } catch (e: InterruptedException) {
                Thread.currentThread().interrupt()
                return androidx.media3.common.C.RESULT_END_OF_INPUT
            }

            currentPosition = 0

            // If a zero-length sentinel was offered to signal completion, treat as EOF
            if (chunk.isEmpty()) {
                return androidx.media3.common.C.RESULT_END_OF_INPUT
            }
        }

        val available = chunk.size - currentPosition
        if (available <= 0) {
            throw IOException("Invalid current position in audio chunk")
        }

        val toCopy = minOf(length, available)
        try {
            System.arraycopy(chunk, currentPosition, target, offset, toCopy)
        } catch (e: Exception) {
            throw IOException("Failed to copy audio data: ${e.message}")
        }
        currentPosition += toCopy
        bytesTransferred(toCopy) // Report successful transfer

        return toCopy
    }

    // Return a stable, non-null URI so wrappers like StatsDataSource don't NPE
    override fun getUri(): android.net.Uri = android.net.Uri.parse("buffered://pcm")

    override fun close() {
        if (opened) {
            opened = false
            transferEnded()
        }
    }
}

@androidx.annotation.OptIn(androidx.media3.common.util.UnstableApi::class)
@Composable
fun StreamPlayerScreen(text: String, serverUrl: String) {
    val context = LocalContext.current
    var player by remember { mutableStateOf<ExoPlayer?>(null) }
    var audioTrackRef by remember { mutableStateOf<AudioTrack?>(null) }
    var isLoading by remember { mutableStateOf(true) }
    var error by remember { mutableStateOf<String?>(null) }
    var isPlaying by remember { mutableStateOf(false) }
    var playbackSpeed by remember { mutableStateOf(1.0f) }
    var streamEnded by remember { mutableStateOf(false) }
    var debugLog by remember { mutableStateOf("Initializing...") }
    var chunksReceived by remember { mutableStateOf(0) }
    var totalChunks by remember { mutableStateOf(0) }
    var highlightedWordIndex by remember { mutableStateOf(0) }
    val words = remember { text.split(" ") }

    val scope = rememberCoroutineScope()
    val audioBuffer = remember { LinkedBlockingQueue<ByteArray>() }
    val streamCompleteFlag = remember { AtomicBoolean(false) }

    // PCM format info (make available across coroutines)
    val pcmFormat = Format.Builder()
        .setSampleMimeType(MimeTypes.AUDIO_RAW)
        .setChannelCount(1)  // mono
        .setSampleRate(22050) // Must match server's sample rate
        .setPcmEncoding(androidx.media3.common.C.ENCODING_PCM_16BIT)
        .build()

    DisposableEffect(Unit) {
        var responseStream: InputStream? = null
        var sessionId: String? = null

        val job = scope.launch {
            // Set up ExoPlayer on main thread
            withContext(Dispatchers.Main) {
                try {
                    val dataSourceFactory = DataSource.Factory {
                        BufferedAudioDataSource(audioBuffer, pcmFormat) { streamCompleteFlag.get() }
                    }


                    // We'll use a low-level AudioTrack for PCM playback instead of ExoPlayer.
                    // Create a shared AudioTrack instance (will be started when first chunk arrives).
                    val sampleRate = pcmFormat.sampleRate
                    val channelMask = AudioFormat.CHANNEL_OUT_MONO
                    val encoding = AudioFormat.ENCODING_PCM_16BIT
                    val minBuf = AudioTrack.getMinBufferSize(sampleRate, channelMask, encoding).coerceAtLeast(8192)
                    debugLog = "AudioTrack buffer size: $minBuf bytes"

                    withContext(Dispatchers.Main) {
                        try {
                            val at = AudioTrack.Builder()
                                .setAudioAttributes(
                                    AudioAttributes.Builder()
                                        .setUsage(AudioAttributes.USAGE_MEDIA)
                                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                                        .build()
                                )
                                .setAudioFormat(
                                    AudioFormat.Builder()
                                        .setEncoding(encoding)
                                        .setChannelMask(channelMask)
                                        .setSampleRate(sampleRate)
                                        .build()
                                )
                                .setBufferSizeInBytes(minBuf)
                                .setTransferMode(AudioTrack.MODE_STREAM)
                                .build()
                            audioTrackRef = at
                        } catch (e: Exception) {
                            error = "AudioTrack init failed: ${e.message}"
                            debugLog = "AudioTrack error: ${e.message}"
                            isLoading = false
                        }
                    }

                } catch (e: Exception) {
                    error = "Player setup failed: ${e.message}"
                    debugLog = "Setup error: ${e.message}"
                    isLoading = false
                }
            }

            // Handle network communication in background
            withContext(Dispatchers.IO) {
                try {
                    debugLog = "Checking server..."
                    val ok = ServerAPI.checkServerStatus(serverUrl)
                    if (!ok) {
                        withContext(Dispatchers.Main) {
                            error = "Unable to reach server at $serverUrl. Check network or server address."
                            isLoading = false
                        }
                        return@withContext
                    }

                    debugLog = "Connecting to server..."
                    val response = try {
                        ServerAPI.streamAudio(text, serverUrl)
                    } catch (e: java.net.UnknownHostException) {
                        withContext(Dispatchers.Main) {
                            error = "DNS lookup failed for server: ${e.message}"
                            debugLog = "DNS lookup failed: ${e.message}"
                            isLoading = false
                        }
                        return@withContext
                    }

                    if (!response.isSuccessful) {
                        throw IOException("Server error: ${response.code}")
                    }

                    responseStream = response.body!!.byteStream()
                    val input = BufferedInputStream(responseStream)
                    val boundary = "--CHUNK_BOUNDARY--".toByteArray()
                    val newline = "\n".toByteArray()

                    debugLog = "Connected, receiving data..."

                    // Manual parser state
                    val bufferStream = ByteArrayOutputStream()
                    val window = ByteArray(boundary.size)
                    var windowPos = 0
                    var readingAudio = false

                    while (isActive) {
                        val b = input.read()
                        if (b == -1) {
                            debugLog = "End of stream reached"
                            break
                        }

                        bufferStream.write(b)
                        window[windowPos] = b.toByte()
                        windowPos = (windowPos + 1) % boundary.size
                        
                        // Log every 1000 bytes for diagnostics
                        if (bufferStream.size() % 1000 == 0) {
                            debugLog = "Reading stream: ${bufferStream.size()} bytes buffered"
                        }

                        // Check for boundary match in circular buffer
                        var match = true
                        for (j in boundary.indices) {
                            val idx = (windowPos + j) % boundary.size
                            if (window[idx] != boundary[j]) {
                                match = false
                                break
                            }
                        }

                        if (match) {
                            val content = bufferStream.toByteArray()
                            // remove trailing boundary bytes
                            val contentWithoutBoundary = if (content.size >= boundary.size) content.copyOf(content.size - boundary.size) else ByteArray(0)
                            bufferStream.reset()

                            if (contentWithoutBoundary.isNotEmpty()) {
                                // Remove leading/trailing newlines around JSON if present
                                val trimmed = trimNewlines(contentWithoutBoundary)

                                if (!readingAudio) {
                                    // Expect JSON metadata
                                    val jsonString = String(trimmed, Charsets.UTF_8).trim()
                                    if (jsonString.isNotEmpty() && jsonString.startsWith("{")) {
                                        try {
                                            val json = JSONObject(jsonString)
                                            val type = json.optString("type", "chunk_metadata")
                                            when (type) {
                                                "init" -> {
                                                    sessionId = json.optString("session_id", sessionId)
                                                    totalChunks = json.optInt("total_chunks", totalChunks)
                                                    debugLog = "Session initialized: $totalChunks chunks expected"
                                                }
                                                "chunk_metadata" -> {
                                                    // switch to reading audio for next part
                                                    readingAudio = true
                                                    val wordIndex = json.optInt("word_index", -1)
                                                    if (wordIndex >= 0) {
                                                        withContext(Dispatchers.Main) {
                                                            highlightedWordIndex = wordIndex
                                                        }
                                                    }
                                                }
                                                "end_of_stream" -> {
                                                    val totalDuration = json.optDouble("total_duration", 0.0)
                                                    val finalChunkCount = json.optInt("total_chunks", totalChunks)
                                                    streamCompleteFlag.set(true)
                                                    debugLog = "Stream complete: $chunksReceived/$finalChunkCount chunks, ${totalDuration.toFloat()}s"
                                                    break
                                                }
                                            }
                                        } catch (e: JSONException) {
                                            debugLog = "JSON parse error: ${e.message}"
                                        }
                                    }
                                } else {
                                    // This trimmed blob should be raw PCM audio bytes
                                    val pcmBytes = trimmed
                                    if (pcmBytes.isNotEmpty()) {
                                        debugLog = "Processing chunk: ${pcmBytes.size} bytes"
                                        // Validate size
                                        if (pcmBytes.size % 2 != 0) {
                                            debugLog = "Invalid PCM size: ${pcmBytes.size}"
                                        } else {
                                            // Ensure AudioTrack is started
                                            withContext(Dispatchers.Main) {
                                                try {
                                                    if (audioTrackRef != null && audioTrackRef?.playState != AudioTrack.PLAYSTATE_PLAYING) {
                                                        audioTrackRef?.play()
                                                        isPlaying = true
                                                    }
                                                } catch (e: Exception) {
                                                    debugLog = "AudioTrack play error: ${e.message}"
                                                }
                                            }

                                            // Write PCM to AudioTrack (blocking write)
                                            try {
                                                val result = audioTrackRef?.write(pcmBytes, 0, pcmBytes.size)
                                                chunksReceived += 1
                                                if (chunksReceived == 1) {
                                                    withContext(Dispatchers.Main) {
                                                        isLoading = false  // Switch to player view on first chunk
                                                    }
                                                }
                                                debugLog = "Wrote ${pcmBytes.size} bytes to AudioTrack, result: $result"
                                            } catch (e: Exception) {
                                                debugLog = "Audio write failed: ${e.message}"
                                            }
                                        }
                                    }
                                    // Reset for next JSON metadata
                                    readingAudio = false
                                }
                            }
                        }
                    }

                    // Close input
                    try { input.close() } catch (e: IOException) {}

                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        error = "Stream failed: ${e.message}"
                        debugLog = "Exception: ${e.message}"
                        isLoading = false
                    }
                    e.printStackTrace()
                } finally {
                    streamCompleteFlag.set(true)
                    // Offer a sentinel so readers unblock
                    try { audioBuffer.offer(ByteArray(0)) } catch (e: Exception) {}
                    try {
                        responseStream?.close()
                    } catch (e: IOException) {}
                    // Stop and release AudioTrack if initialized
                    try {
                        withContext(Dispatchers.Main) {
                            audioTrackRef?.let { at ->
                                try { at.stop() } catch (_: Exception) {}
                                try { at.release() } catch (_: Exception) {}
                            }
                            audioTrackRef = null
                        }
                    } catch (e: Exception) {
                        // ignore
                    }
                }
            }
        }

        onDispose {
            job.cancel()
            try {
                responseStream?.close()
            } catch (e: IOException) {}
            player?.release()
            // Release any AudioTrack if present
            try {
                // We constructed AudioTrack inside the coroutine; attempt a safe release via reflection of local var not possible here.
                // Best-effort: nothing else to do since audioTrack is scoped inside coroutine. If needed we can promote it to outer scope.
            } catch (e: Exception) {}
        }

        // Return DisposableEffectResult
        onDispose {}
    }

    LaunchedEffect(playbackSpeed, player) {
        player?.let {
            val params = PlaybackParameters(playbackSpeed)
            it.playbackParameters = params
        }
    }

    Scaffold { padding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
            contentAlignment = Alignment.Center
        ) {
            when {
                isLoading -> {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator()
                        Spacer(modifier = Modifier.height(16.dp))
                        Text("Connecting...")
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            debugLog,
                            style = MaterialTheme.typography.bodySmall,
                            textAlign = TextAlign.Center
                        )
                    }
                }
                error != null -> {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Text(
                            text = error!!,
                            color = MaterialTheme.colorScheme.error,
                            textAlign = TextAlign.Center
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = debugLog,
                            style = MaterialTheme.typography.bodySmall,
                            textAlign = TextAlign.Center
                        )
                    }
                }
                (!isLoading && error == null) -> {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(16.dp)
                    ) {
                        Text(
                            text = buildAnnotatedString {
                                words.forEachIndexed { idx, word ->
                                    if (idx == highlightedWordIndex) {
                                        withStyle(SpanStyle(fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)) {
                                            append("$word ")
                                        }
                                    } else {
                                        append("$word ")
                                    }
                                }
                            },
                            modifier = Modifier
                                .weight(1f)
                                .fillMaxWidth()
                                .verticalScroll(rememberScrollState())
                                .padding(bottom = 16.dp),
                            lineHeight = 24.sp
                        )

                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(
                                containerColor = MaterialTheme.colorScheme.surfaceVariant
                            )
                        ) {
                            Column(modifier = Modifier.padding(8.dp)) {
                                Text(
                                    "Status: $debugLog",
                                    style = MaterialTheme.typography.bodySmall
                                )
                                if (totalChunks > 0) {
                                    LinearProgressIndicator(
                                        progress = chunksReceived.toFloat() / totalChunks.toFloat(),
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(vertical = 4.dp)
                                    )
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(16.dp))

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.Center,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            // Back button
                            IconButton(
                                onClick = {
                                    highlightedWordIndex = (highlightedWordIndex - 1).coerceAtLeast(0)
                                }
                            ) {
                                Icon(
                                    imageVector = Icons.Default.PlayArrow,
                                    contentDescription = "Previous Word",
                                    modifier = Modifier.rotate(180f)
                                )
                            }
                            
                            Spacer(modifier = Modifier.width(16.dp))
                            
                            // Play/Pause button
                            IconButton(
                                onClick = {
                                    if (isPlaying) {
                                        audioTrackRef?.pause()
                                        isPlaying = false
                                    } else {
                                        audioTrackRef?.play()
                                        isPlaying = true
                                    }
                                },
                                modifier = Modifier.size(64.dp)
                            ) {
                                Icon(
                                    imageVector = if (isPlaying) Icons.Default.Pause else Icons.Default.PlayArrow,
                                    contentDescription = "Play/Pause",
                                    modifier = Modifier.fillMaxSize()
                                )
                            }
                            
                            Spacer(modifier = Modifier.width(16.dp))
                            
                            // Forward button
                            IconButton(
                                onClick = {
                                    highlightedWordIndex = (highlightedWordIndex + 1).coerceAtMost(words.size - 1)
                                }
                            ) {
                                Icon(
                                    imageVector = Icons.Default.PlayArrow,
                                    contentDescription = "Next Word"
                                )
                            }

                            Spacer(modifier = Modifier.height(16.dp))

                            Text("Playback Speed: ${(playbackSpeed * 100).roundToInt() / 100f}x")
                            Slider(
                                value = playbackSpeed,
                                onValueChange = { newSpeed -> playbackSpeed = newSpeed },
                                valueRange = 0.5f..2.0f,
                                steps = 5
                            )
                        }
                    }
                }
            }
        }
    }
}

// Trim leading/trailing newline characters (\n or \r) from a byte array
private fun trimNewlines(bytes: ByteArray): ByteArray {
    var start = 0
    var end = bytes.size
    while (start < end && (bytes[start] == '\n'.code.toByte() || bytes[start] == '\r'.code.toByte())) start++
    while (end > start && (bytes[end - 1] == '\n'.code.toByte() || bytes[end - 1] == '\r'.code.toByte())) end--
    return if (start == 0 && end == bytes.size) bytes else bytes.copyOfRange(start, end)
}