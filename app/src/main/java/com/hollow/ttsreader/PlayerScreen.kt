package com.hollow.ttsreader

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.common.MediaItem
import androidx.media3.common.MimeTypes
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.common.util.UnstableApi
import androidx.media3.datasource.DataSource
import androidx.media3.datasource.okhttp.OkHttpDataSource
import androidx.media3.datasource.setPostRequest // The crucial import
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.source.ProgressiveMediaSource
import androidx.media3.ui.PlayerView
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import okhttp3.OkHttpClient

@Composable
fun PlayerScreen(book: Book) {
    val context = LocalContext.current
    val player = remember {
        ExoPlayer.Builder(context).build().apply {
            setMediaItem(MediaItem.fromUri(book.audioPath))
            prepare()
            playWhenReady = true
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            player.release()
        }
    }

    AndroidView(factory = { PlayerView(it).apply { this.player = player } })
}

@androidx.annotation.OptIn(UnstableApi::class)
@Composable
fun StreamPlayerScreen(text: String, serverUrl: String) {
    val context = LocalContext.current
    var playerState by remember { mutableStateOf<Player?>(null) }
    var isLoading by remember { mutableStateOf(true) }
    var error by remember { mutableStateOf<String?>(null) }

    val exoPlayer = remember {
        val client = OkHttpClient.Builder().build()
        val dataSourceFactory: DataSource.Factory = OkHttpDataSource.Factory(client)

        ExoPlayer.Builder(context)
            .setMediaSourceFactory(ProgressiveMediaSource.Factory(dataSourceFactory))
            .build()
    }

    LaunchedEffect(serverUrl, text) {
        val streamUrl = "$serverUrl/stream"
        val jsonPayload = buildJsonObject { put("text", text) }.toString()

        val mediaItem = MediaItem.Builder()
            .setUri(streamUrl)
            .setMimeType(MimeTypes.AUDIO_WAV)
            .setPostRequest(jsonPayload.toByteArray(), "application/json")
            .build()

        exoPlayer.setMediaItem(mediaItem)
        exoPlayer.prepare()
        exoPlayer.playWhenReady = true

        val listener = object : Player.Listener {
            override fun onIsLoadingChanged(loading: Boolean) {
                if (playerState == null) {
                    isLoading = loading
                } else if (!loading) {
                    isLoading = false
                }
            }

            override fun onPlayerError(e: PlaybackException) {
                error = "Playback failed: ${e.message} (Code: ${e.errorCodeName})"
                isLoading = false
            }
        }

        exoPlayer.addListener(listener)
        playerState = exoPlayer
    }

    DisposableEffect(Unit) {
        onDispose {
            playerState?.release()
        }
    }

    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        when {
            error != null -> {
                Text(
                    text = error!!,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.align(Alignment.Center)
                )
            }
            playerState != null -> {
                AndroidView(
                    factory = { ctx -> PlayerView(ctx).apply { this.player = playerState } },
                    modifier = Modifier.fillMaxSize()
                )
            }
            else -> {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CircularProgressIndicator()
                    Spacer(modifier = Modifier.height(16.dp))
                    Text("Connecting to stream...")
                }
            }
        }
    }
}
