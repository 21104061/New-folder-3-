package com.hollow.ttsreader

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.media3.common.MediaItem
import androidx.media3.common.Player
import androidx.media3.exoplayer.ExoPlayer
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive

@androidx.annotation.OptIn(androidx.media3.common.util.UnstableApi::class)
@OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)
@Composable
fun PlayerScreen(book: Book) {
    val context = LocalContext.current
    
    // Load timestamps and text
    val timestamps = remember { book.getTimestamps() }
    val bookText = remember { book.getText() }
    val words = remember { bookText.split(Regex("\\s+")) }
    
    // State variables
    var isPlaying by remember { mutableStateOf(false) }
    var currentPosition by remember { mutableStateOf(0L) }
    var duration by remember { mutableStateOf(0L) }
    var highlightedWordIndex by remember { mutableStateOf(-1) }
    var playbackSpeed by remember { mutableStateOf(1.0f) }
    
    // Create ExoPlayer
    val player = remember {
        ExoPlayer.Builder(context).build().apply {
            setMediaItem(MediaItem.fromUri(book.audioPath))
            prepare()
            
            addListener(object : Player.Listener {
                override fun onIsPlayingChanged(playing: Boolean) {
                    isPlaying = playing
                }
            })
        }
    }
    
    // Update playback position and word highlighting
    LaunchedEffect(player) {
        while (isActive) {
            currentPosition = player.currentPosition
            duration = player.duration.coerceAtLeast(0L)
            
            // Find current word based on timestamp
            if (timestamps.isNotEmpty()) {
                val currentSeconds = currentPosition / 1000.0
                val index = timestamps.indexOfFirst { 
                    currentSeconds >= it.start && currentSeconds <= it.end 
                }
                if (index != -1) {
                    highlightedWordIndex = index
                }
            }
            
            delay(100)
        }
    }
    
    // Update playback speed
    LaunchedEffect(playbackSpeed) {
        player.setPlaybackSpeed(playbackSpeed)
    }
    
    DisposableEffect(Unit) {
        onDispose {
            player.release()
        }
    }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        // Text display with word highlighting
        if (timestamps.isNotEmpty() && words.isNotEmpty()) {
            Text(
                text = buildAnnotatedString {
                    timestamps.forEachIndexed { idx, timestamp ->
                        if (idx == highlightedWordIndex) {
                            withStyle(
                                SpanStyle(
                                    fontWeight = FontWeight.Bold,
                                    color = MaterialTheme.colorScheme.primary,
                                    fontSize = 18.sp
                                )
                            ) {
                                append("${timestamp.word} ")
                            }
                        } else {
                            append("${timestamp.word} ")
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
        } else {
            // Fallback if no timestamps
            Text(
                text = bookText,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
                    .padding(bottom = 16.dp),
                lineHeight = 24.sp
            )
        }
        
        // Playback controls card
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            )
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                // Book title
                Text(
                    text = book.title,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                // Time display
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = formatMilliseconds(currentPosition),
                        style = MaterialTheme.typography.bodyMedium
                    )
                    Text(
                        text = formatMilliseconds(duration),
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
                
                // Seek bar
                Slider(
                    value = if (duration > 0) currentPosition.toFloat() else 0f,
                    onValueChange = { player.seekTo(it.toLong()) },
                    valueRange = 0f..(duration.toFloat().coerceAtLeast(1f)),
                    modifier = Modifier.fillMaxWidth()
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                // Playback controls
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // Rewind 10s
                    IconButton(onClick = { 
                        player.seekTo((currentPosition - 10000).coerceAtLeast(0))
                    }) {
                        Icon(Icons.Default.Replay, "Rewind 10s")
                    }
                    
                    // Play/Pause
                    IconButton(
                        onClick = {
                            if (isPlaying) player.pause() else player.play()
                        },
                        modifier = Modifier.size(64.dp)
                    ) {
                        Icon(
                            imageVector = if (isPlaying) Icons.Default.Pause else Icons.Default.PlayArrow,
                            contentDescription = "Play/Pause",
                            modifier = Modifier.size(48.dp)
                        )
                    }
                    
                    // Forward 10s
                    IconButton(onClick = { 
                        player.seekTo((currentPosition + 10000).coerceAtMost(duration))
                    }) {
                        Icon(Icons.Default.Forward10, "Forward 10s")
                    }
                }
                
                Spacer(modifier = Modifier.height(8.dp))
                
                // Playback speed control
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Speed: ${String.format("%.1f", playbackSpeed)}x",
                        style = MaterialTheme.typography.bodyMedium
                    )
                    Row {
                        listOf(0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 2.0f).forEach { speed ->
                            FilterChip(
                                selected = playbackSpeed == speed,
                                onClick = { playbackSpeed = speed },
                                label = { Text("${speed}x") },
                                modifier = Modifier.padding(horizontal = 2.dp)
                            )
                        }
                    }
                }
                
                // Info
                if (timestamps.isNotEmpty()) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Word sync: ${timestamps.size} words tracked",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

// Format milliseconds to MM:SS or HH:MM:SS
private fun formatMilliseconds(millis: Long): String {
    val totalSeconds = (millis / 1000).toInt()
    val hours = totalSeconds / 3600
    val minutes = (totalSeconds % 3600) / 60
    val secs = totalSeconds % 60
    
    return if (hours > 0) {
        String.format("%d:%02d:%02d", hours, minutes, secs)
    } else {
        String.format("%02d:%02d", minutes, secs)
    }
}
