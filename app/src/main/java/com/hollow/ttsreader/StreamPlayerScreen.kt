package com.hollow.ttsreader

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.media3.common.MediaItem
import androidx.media3.common.MimeTypes
import androidx.media3.common.PlaybackParameters
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
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
import java.io.PipedInputStream
import java.io.PipedOutputStream
import java.util.LinkedList
import kotlin.math.roundToInt

// Helper function to read from the stream until a boundary is found
private fun readUntilBoundary(inputStream: InputStream, boundary: ByteArray): ByteArray? {
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
}

@Composable
fun StreamPlayerScreen(text: String, serverUrl: String) {
    val context = LocalContext.current
    var player by remember { mutableStateOf<ExoPlayer?>(null) }
    var isLoading by remember { mutableStateOf(true) }
    var error by remember { mutableStateOf<String?>(null) }
    var isPlaying by remember { mutableStateOf(false) }
    var playbackSpeed by remember { mutableStateOf(1.0f) }
    val allTimestamps = remember { mutableStateListOf<Timestamp>() }
    var highlightedWordIndex by remember { mutableStateOf(-1) }
    var streamEnded by remember { mutableStateOf(false) }

    val scope = rememberCoroutineScope()
    val words = remember { text.split(" ") }

    DisposableEffect(Unit) {
        val pipeWriter = PipedOutputStream()
        val pipeReader = PipedInputStream(pipeWriter)
        var responseStream: InputStream? = null

        val job = scope.launch {
            // On the main thread, set up ExoPlayer to read from the pipe
            withContext(Dispatchers.Main) {
                val dataSourceFactory = DataSource.Factory { InputStreamDataSource(pipeReader) }
                val mediaItem = MediaItem.Builder()
                    .setUri("piped://audio.wav")
                    .setMimeType(MimeTypes.AUDIO_WAV)
                    .build()
                val mediaSource = ProgressiveMediaSource.Factory(dataSourceFactory)
                    .createMediaSource(mediaItem)

                val newPlayer = ExoPlayer.Builder(context).build().apply {
                    setMediaSource(mediaSource)
                    prepare()
                    playWhenReady = true
                    addListener(object : Player.Listener {
                        override fun onPlaybackStateChanged(playbackState: Int) {
                            if (playbackState == Player.STATE_READY && isLoading) {
                                isLoading = false
                            }
                            if (playbackState == Player.STATE_ENDED) {
                                streamEnded = true
                            }
                        }

                        override fun onIsPlayingChanged(playing: Boolean) {
                            isPlaying = playing
                        }

                        override fun onPlayerError(e: PlaybackException) {
                            error = "Playback failed: ${e.message}"
                            isLoading = false
                        }
                    })
                }
                player = newPlayer
            }

            // In a background thread, handle network communication
            withContext(Dispatchers.IO) {
                var sessionId: String? = null
                try {
                    // Write a WAV header to the pipe first, so ExoPlayer knows how to play the raw PCM audio
                    val sampleRate = 22050
                    val channels = 1
                    val bitDepth = 16
                    val byteRate = sampleRate * channels * bitDepth / 8
                    val blockAlign = channels * bitDepth / 8
                    val header = ByteArray(44)

                    header[0] = 'R'.code.toByte(); header[1] = 'I'.code.toByte(); header[2] = 'F'.code.toByte(); header[3] = 'F'.code.toByte()
                    header[4] = 0; header[5] = 0; header[6] = 0; header[7] = 0
                    header[8] = 'W'.code.toByte(); header[9] = 'A'.code.toByte(); header[10] = 'V'.code.toByte(); header[11] = 'E'.code.toByte()
                    header[12] = 'f'.code.toByte(); header[13] = 'm'.code.toByte(); header[14] = 't'.code.toByte(); header[15] = ' '.code.toByte()
                    header[16] = 16; header[17] = 0; header[18] = 0; header[19] = 0
                    header[20] = 1; header[21] = 0
                    header[22] = channels.toByte(); header[23] = 0
                    header[24] = (sampleRate and 0xff).toByte()
                    header[25] = ((sampleRate shr 8) and 0xff).toByte()
                    header[26] = ((sampleRate shr 16) and 0xff).toByte()
                    header[27] = ((sampleRate shr 24) and 0xff).toByte()
                    header[28] = (byteRate and 0xff).toByte()
                    header[29] = ((byteRate shr 8) and 0xff).toByte()
                    header[30] = ((byteRate shr 16) and 0xff).toByte()
                    header[31] = ((byteRate shr 24) and 0xff).toByte()
                    header[32] = blockAlign.toByte(); header[33] = 0
                    header[34] = bitDepth.toByte(); header[35] = 0
                    header[36] = 'd'.code.toByte(); header[37] = 'a'.code.toByte(); header[38] = 't'.code.toByte(); header[39] = 'a'.code.toByte()
                    header[40] = 0; header[41] = 0; header[42] = 0; header[43] = 0
                    pipeWriter.write(header)


                    val response = ServerAPI.streamAudio(text, serverUrl)
                    if (!response.isSuccessful) {
                        throw IOException("Server error: ${response.code}")
                    }

                    responseStream = response.body!!.byteStream()
                    val boundary = "--CHUNK_BOUNDARY--".toByteArray()

                    while (isActive) {
                        val partData = readUntilBoundary(responseStream!!, boundary) ?: break
                        if (partData.isEmpty()) {
                            continue
                        }

                        // Find the first newline character
                        val newlineIndex = partData.indexOf(10.toByte())

                        val jsonBytes: ByteArray
                        val audioData: ByteArray?

                        if (newlineIndex != -1) {
                            jsonBytes = partData.copyOfRange(0, newlineIndex)
                            audioData = partData.copyOfRange(newlineIndex + 1, partData.size)
                        } else {
                            // Assume the whole part is JSON if no newline is found
                            jsonBytes = partData
                            audioData = null
                        }

                        try {
                            val jsonString = String(jsonBytes, Charsets.UTF_8).trim()
                            if (jsonString.isEmpty()) continue

                            val json = JSONObject(jsonString)

                            when (json.optString("type")) {
                                "init" -> {
                                    sessionId = json.getString("session_id")
                                    launch { // Fire-and-forget ACK
                                        ServerAPI.sendChunkReceivedConfirmation(serverUrl, sessionId!!, 0)
                                    }
                                }
                                "chunk_metadata" -> {
                                    val chunkIndex = json.getInt("chunk_index")
                                    if (sessionId != null) {
                                        if (audioData != null && audioData.isNotEmpty()) {
                                            pipeWriter.write(audioData)
                                            pipeWriter.flush()
                                        }
                                        launch { // Fire-and-forget ACK
                                            ServerAPI.sendChunkReceivedConfirmation(serverUrl, sessionId, chunkIndex)
                                        }
                                    } else {
                                        throw IOException("Received audio chunk before session initialization.")
                                    }
                                }
                                "timestamp" -> {
                                    val word = json.getString("word")
                                    val startTime = json.getDouble("start_time")
                                    val endTime = json.getDouble("end_time")
                                    withContext(Dispatchers.Main) {
                                        allTimestamps.add(Timestamp(word, startTime, endTime))
                                    }
                                }
                                "end_of_stream" -> {
                                    break // Graceful end
                                }
                            }
                        } catch (e: JSONException) {
                            println("Warning: JSON parsing failed: ${e.message}")
                        }
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        error = "Stream failed: ${e.message}"
                        isLoading = false
                    }
                } finally {
                    try {
                        responseStream?.close()
                        pipeWriter.close()
                    } catch (e: IOException) {}
                }
            }
        }

        onDispose {
            job.cancel()
            try {
                pipeReader.close()
                pipeWriter.close()
                responseStream?.close()
            } catch (e: IOException) {}
            player?.release()
        }
    }


    LaunchedEffect(playbackSpeed, player) {
        player?.let {
            val params = PlaybackParameters(playbackSpeed)
            it.playbackParameters = params
        }
    }

    LaunchedEffect(player, isPlaying) {
        if (isPlaying) {
            while (isActive) {
                val currentPosition = player?.currentPosition ?: 0L
                val currentIndex = allTimestamps.indexOfLast {
                    val startTimeMs = it.startTime * 1000
                    val endTimeMs = it.endTime * 1000
                    currentPosition in startTimeMs.toLong()..endTimeMs.toLong()
                }
                if (highlightedWordIndex != currentIndex) {
                     withContext(Dispatchers.Main) {
                        highlightedWordIndex = currentIndex
                    }
                }
                delay(100)
            }
        }
    }

    Scaffold {
        padding ->
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
                        Text("Connecting to stream...")
                    }
                }
                error != null -> {
                    Text(
                        text = error!!,
                        color = MaterialTheme.colorScheme.error,
                        modifier = Modifier.padding(16.dp),
                        textAlign = TextAlign.Center
                    )
                }
                player != null -> {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(16.dp)
                    ) {
                        Text(
                            text = buildAnnotatedString {
                                words.forEachIndexed { index, word ->
                                    if (index == highlightedWordIndex) {
                                        withStyle(style = SpanStyle(fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)) {
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

                        if (streamEnded) {
                            Text(
                                "End of stream",
                                modifier = Modifier.align(Alignment.CenterHorizontally),
                                style = MaterialTheme.typography.bodySmall
                            )
                        }

                        Column(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            IconButton(
                                onClick = {
                                    if (isPlaying) player?.pause() else player?.play()
                                },
                                modifier = Modifier.size(64.dp)
                            ) {
                                Icon(
                                    imageVector = if (isPlaying) Icons.Default.Pause else Icons.Default.PlayArrow,
                                    contentDescription = "Play/Pause",
                                    modifier = Modifier.fillMaxSize()
                                )
                            }
                            Spacer(modifier = Modifier.height(24.dp))

                            Text("Speed: ${(playbackSpeed * 100).roundToInt() / 100f}x")
                            Slider(
                                value = playbackSpeed,
                                onValueChange = { newSpeed ->
                                    playbackSpeed = newSpeed
                                },
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
