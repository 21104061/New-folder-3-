package com.hollow.ttsreader

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.Info
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.workDataOf
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.UUID

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            TTSReaderTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    TTSReaderApp()
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TTSReaderApp() {
    val context = LocalContext.current
    val bookManager = remember { BookManager(context) }
    val appPrefs = remember { AppPreferences(context) }
    var books by remember { mutableStateOf(bookManager.getBooks()) }

    var currentScreen by remember {
        mutableStateOf<Screen>(
            if (appPrefs.isFirstLaunch || appPrefs.serverUrl.isEmpty()) {
                Screen.Settings
            } else {
                Screen.Library
            }
        )
    }
    var selectedBook by remember { mutableStateOf<Book?>(null) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(when (val screen = currentScreen) {
                        Screen.Library -> "My Books"
                        Screen.Upload -> "Convert New Book"
                        Screen.Settings -> "Server Settings"
                        is Screen.Player -> screen.book.title
                        is Screen.StreamPlayer -> "Listening"
                    })
                },
                navigationIcon = {
                    if (currentScreen !is Screen.Library && !(currentScreen is Screen.Settings && appPrefs.isFirstLaunch)) {
                        IconButton(onClick = { currentScreen = Screen.Library }) {
                            Icon(Icons.Default.ArrowBack, "Back")
                        }
                    }
                },
                actions = {
                    if (currentScreen == Screen.Library) {
                        IconButton(onClick = { currentScreen = Screen.Settings }) {
                            Icon(Icons.Default.Settings, "Settings")
                        }
                    }
                    if (currentScreen is Screen.Player) {
                        IconButton(onClick = {
                            val bookToDelete = (currentScreen as Screen.Player).book
                            bookManager.deleteBook(bookToDelete.id)
                            books = bookManager.getBooks()
                            currentScreen = Screen.Library
                        }) {
                            Icon(Icons.Default.Delete, "Delete Book")
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            )
        },
        floatingActionButton = {
            if (currentScreen == Screen.Library) {
                FloatingActionButton(onClick = { currentScreen = Screen.Upload }) {
                    Icon(Icons.Default.Add, "Add Book")
                }
            }
        }
    ) { padding ->
        Box(modifier = Modifier.padding(padding)) {
            when (val screen = currentScreen) {
                Screen.Library -> {
                    LibraryScreen(
                        books = books,
                        onBookClick = { book ->
                            selectedBook = book
                            currentScreen = Screen.Player(book)
                        },
                        onRefresh = { books = bookManager.getBooks() }
                    )
                }
                Screen.Upload -> {
                    UploadScreen(
                        appPrefs = appPrefs,
                        onBookConverted = { book ->
                            bookManager.saveBook(book)
                            books = bookManager.getBooks()
                            currentScreen = Screen.Library
                        },
                        onStream = { text ->
                            currentScreen = Screen.StreamPlayer(text)
                        },
                        onCancel = { currentScreen = Screen.Library }
                    )
                }
                Screen.Settings -> {
                    SettingsScreen(
                        appPrefs = appPrefs,
                        onSaved = {
                            if (appPrefs.isFirstLaunch) appPrefs.isFirstLaunch = false
                            currentScreen = Screen.Library
                        }
                    )
                }
                is Screen.Player -> {
                    PlayerScreen(book = screen.book)
                }
                is Screen.StreamPlayer -> {
                    StreamPlayerScreen(text = screen.text, serverUrl = appPrefs.serverUrl)
                }
            }
        }
    }
}

@Composable
fun UploadScreen(
    appPrefs: AppPreferences,
    onBookConverted: (Book) -> Unit,
    onStream: (String) -> Unit,
    onCancel: () -> Unit
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var textState by remember { mutableStateOf(TextFieldValue("")) }
    var titleState by remember { mutableStateOf("") }
    var isLoading by remember { mutableStateOf(false) }
    var loadingMessage by remember { mutableStateOf("Converting book...") }
    var showEmptyError by remember { mutableStateOf(false) }
    var showError by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf("") }

    val filePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            val fileName = it.path?.substringAfterLast('/') ?: "Untitled"
            titleState = fileName.substringBeforeLast('.')
            context.contentResolver.openInputStream(it)?.use { stream ->
                textState = TextFieldValue(stream.reader().readText())
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        if (isLoading) {
            Box(contentAlignment = Alignment.Center, modifier = Modifier.fillMaxSize()) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CircularProgressIndicator()
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(loadingMessage, style = MaterialTheme.typography.bodyLarge)
                }
            }
        } else if (showError) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Icon(
                    Icons.Default.Error,
                    contentDescription = null,
                    modifier = Modifier.size(64.dp),
                    tint = MaterialTheme.colorScheme.error
                )
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    "Conversion Failed",
                    style = MaterialTheme.typography.titleLarge,
                    color = MaterialTheme.colorScheme.error
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    errorMessage,
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(24.dp))
                Button(onClick = {
                    showError = false
                    errorMessage = ""
                }) {
                    Text("Try Again")
                }
            }
        } else {
            OutlinedTextField(
                value = titleState,
                onValueChange = { titleState = it },
                label = { Text("Book Title") },
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = textState,
                onValueChange = { textState = it },
                label = { Text("Book Text") },
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
            )
            if (showEmptyError) {
                Text("Title and text cannot be empty.", color = MaterialTheme.colorScheme.error)
            }
            Spacer(modifier = Modifier.height(16.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = { filePicker.launch("text/plain") }, modifier = Modifier.weight(1f)) {
                    Text("Open .txt")
                }
                Button(onClick = { onCancel() }, modifier = Modifier.weight(1f)) {
                    Text("Cancel")
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                // Stream Button
                Button(
                    onClick = {
                        if (titleState.isNotBlank() && textState.text.isNotBlank()) {
                            showEmptyError = false
                            onStream(textState.text)
                        } else {
                            showEmptyError = true
                        }
                    },
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Stream Audio")
                }

                // Download Button - FIXED VERSION
                Button(
                    onClick = {
                        if (titleState.isNotBlank() && textState.text.isNotBlank()) {
                            showEmptyError = false

                            scope.launch {
                                try {
                                    isLoading = true
                                    loadingMessage = "Validating server connection..."

                                    // First, check if server is reachable
                                    val serverReachable = withContext(Dispatchers.IO) {
                                        ServerAPI.checkServerStatus(appPrefs.serverUrl)
                                    }

                                    if (!serverReachable) {
                                        isLoading = false
                                        showError = true
                                        errorMessage = "Cannot reach server at ${appPrefs.serverUrl}. Please check your server URL in Settings and ensure the Colab notebook is running."
                                        return@launch
                                    }

                                    loadingMessage = "Server connected. Starting conversion..."

                                    // Create book immediately with CONVERTING status
                                    val bookId = UUID.randomUUID().toString()
                                    val bookManager = BookManager(context)
                                    val bookDir = bookManager.createBookDirectory(bookId)

                                    // Save text file immediately
                                    val textFile = File(bookDir, "book_text.txt")
                                    textFile.writeText(textState.text)

                                    // Calculate word count
                                    val wordCount = textState.text.split(Regex("\\s+")).size

                                    // Create placeholder book with CONVERTING status
                                    val placeholderBook = Book(
                                        id = bookId,
                                        title = titleState,
                                        audioPath = "",
                                        timestampsPath = "",
                                        textPath = textFile.absolutePath,
                                        wordCount = wordCount,
                                        duration = "",
                                        status = BookStatus.CONVERTING,
                                        jobId = null // Will be set by worker
                                    )

                                    // Save to library
                                    bookManager.saveBook(placeholderBook)

                                    loadingMessage = "Book added to library. Starting background conversion..."

                                    // Schedule background conversion
                                    val workRequest = OneTimeWorkRequestBuilder<ConversionWorker>()
                                        .setInputData(
                                            workDataOf(
                                                ConversionWorker.KEY_BOOK_ID to bookId,
                                                ConversionWorker.KEY_BOOK_TITLE to titleState,
                                                ConversionWorker.KEY_SERVER_URL to appPrefs.serverUrl,
                                                ConversionWorker.KEY_WORD_COUNT to wordCount
                                            )
                                        )
                                        .build()

                                    WorkManager.getInstance(context).enqueue(workRequest)

                                    // Small delay to show the message
                                    delay(1000)

                                    // Navigate back to library
                                    isLoading = false
                                    onCancel()

                                } catch (e: Exception) {
                                    isLoading = false
                                    showError = true
                                    errorMessage = "Failed to start conversion: ${e.message ?: "Unknown error"}"
                                    println("Error starting conversion: ${e.message}")
                                    e.printStackTrace()
                                }
                            }
                        } else {
                            showEmptyError = true
                        }
                    },
                    modifier = Modifier.weight(1f),
                    enabled = !isLoading
                ) {
                    if (isLoading) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(20.dp),
                            strokeWidth = 2.dp
                        )
                    } else {
                        Text("Download & Save")
                    }
                }
            }
        }
    }
}

@Composable
fun LibraryScreen(
    books: List<Book>,
    onBookClick: (Book) -> Unit,
    onRefresh: () -> Unit
) {
    val context = LocalContext.current
    val bookManager = remember { BookManager(context) }
    var selectionMode by remember { mutableStateOf(false) }
    var selectedIds by remember { mutableStateOf(setOf<String>()) }
    var deletedBooks by remember { mutableStateOf<List<Book>>(emptyList()) }

    // Auto-refresh every 2 seconds if there are converting books
    val hasConvertingBooks = books.any {
        it.status == BookStatus.CONVERTING || it.status == BookStatus.DOWNLOADING
    }

    LaunchedEffect(hasConvertingBooks) {
        if (hasConvertingBooks) {
            while (true) {
                delay(2000)
                onRefresh()
            }
        }
    }

    // Auto-cleanup stale books every 30 seconds
    LaunchedEffect(Unit) {
        while (true) {
            val staleBooks = bookManager.cleanupStaleBooks(2 * 60 * 60 * 1000) // 2 hours
            if (staleBooks.isNotEmpty()) {
                deletedBooks = staleBooks
                onRefresh()
            }
            delay(30000) // Check every 30 seconds
        }
    }

    LaunchedEffect(Unit) {
        onRefresh()
    }

    Column(modifier = Modifier.fillMaxSize()) {
        // Show deleted books report if any
        if (deletedBooks.isNotEmpty()) {
            DeletedBooksReport(deletedBooks) { deletedBooks = emptyList() }
        }

        // Selection controls
        if (books.isNotEmpty()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                AssistChip(
                    onClick = { 
                        selectionMode = !selectionMode
                        if (!selectionMode) selectedIds = emptySet()
                    },
                    label = { Text(if (selectionMode) "Done Selecting" else "Select") },
                    leadingIcon = { Icon(Icons.Default.Checklist, contentDescription = null) }
                )
                
                AssistChip(
                    onClick = {
                        selectionMode = true
                        selectedIds = books.map { it.id }.toSet()
                    },
                    label = { Text("Select All") },
                    leadingIcon = { Icon(Icons.Default.SelectAll, contentDescription = null) }
                )
                
                val canDelete = selectedIds.isNotEmpty()
                AssistChip(
                    onClick = {
                        if (canDelete) {
                            selectedIds.forEach { bookManager.deleteBook(it) }
                            selectedIds = emptySet()
                            selectionMode = false
                            onRefresh()
                        }
                    },
                    enabled = canDelete,
                    label = { Text("Delete Selected") },
                    leadingIcon = { Icon(Icons.Default.Delete, contentDescription = null) }
                )
            }
        }

        if (books.isEmpty()) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Icon(
                        Icons.Default.MenuBook,
                        contentDescription = null,
                        modifier = Modifier.size(64.dp),
                        tint = MaterialTheme.colorScheme.primary
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                    Text("No books yet", style = MaterialTheme.typography.titleLarge)
                    Text("Tap + to convert your first book")
                }
            }
        } else {
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                items(books) { book ->
                    val selected = selectedIds.contains(book.id)
                    BookCard(
                        book = book,
                        onClick = {
                            if (selectionMode) {
                                selectedIds = if (selected) selectedIds - book.id else selectedIds + book.id
                            } else if (book.status == BookStatus.READY) {
                                onBookClick(book)
                            }
                        },
                        showCheckbox = selectionMode,
                        selected = selected,
                        onDelete = {
                            bookManager.deleteBook(book.id)
                            onRefresh()
                        }
                    )
                }
            }
        }
    }
}

@Composable
fun BookCard(
    book: Book, 
    onClick: () -> Unit,
    showCheckbox: Boolean = false,
    selected: Boolean = false,
    onDelete: () -> Unit = {}
) {
    val isClickable = book.status == BookStatus.READY

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(enabled = isClickable, onClick = onClick),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
        colors = CardDefaults.cardColors(
            containerColor = when (book.status) {
                BookStatus.CONVERTING, BookStatus.DOWNLOADING ->
                    MaterialTheme.colorScheme.surfaceVariant
                BookStatus.ERROR ->
                    MaterialTheme.colorScheme.errorContainer
                else ->
                    MaterialTheme.colorScheme.surface
            }
        )
    ) {
        Row(
            modifier = Modifier
                .padding(16.dp)
                .fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Checkbox for selection mode
            if (showCheckbox) {
                Checkbox(
                    checked = selected,
                    onCheckedChange = { onClick() }
                )
                Spacer(modifier = Modifier.width(8.dp))
            }
            
            // Icon based on status
            when (book.status) {
                BookStatus.CONVERTING, BookStatus.DOWNLOADING -> {
                    CircularProgressIndicator(
                        modifier = Modifier.size(40.dp),
                        strokeWidth = 3.dp
                    )
                }
                BookStatus.ERROR -> {
                    Icon(
                        Icons.Default.Error,
                        contentDescription = null,
                        modifier = Modifier.size(40.dp),
                        tint = MaterialTheme.colorScheme.error
                    )
                }
                else -> {
                    Icon(
                        Icons.Default.AutoStories,
                        contentDescription = null,
                        modifier = Modifier.size(40.dp),
                        tint = MaterialTheme.colorScheme.primary
                    )
                }
            }

            Spacer(modifier = Modifier.width(16.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = book.title,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )
                Spacer(modifier = Modifier.height(4.dp))

                val statusText = when (book.status) {
                    BookStatus.CONVERTING -> "Converting..."
                    BookStatus.DOWNLOADING -> "Downloading..."
                    BookStatus.ERROR -> "Conversion failed"
                    BookStatus.READY -> "${book.wordCount} words • ${book.duration.ifBlank { "N/A" }}"
                }

                Text(
                    text = statusText,
                    style = MaterialTheme.typography.bodySmall,
                    color = when (book.status) {
                        BookStatus.ERROR -> MaterialTheme.colorScheme.error
                        else -> MaterialTheme.colorScheme.onSurfaceVariant
                    }
                )
            }

            // Delete button (only show when not in selection mode)
            if (!showCheckbox) {
                IconButton(onClick = onDelete) {
                    Icon(Icons.Default.Delete, contentDescription = "Delete Book")
                }
            }

            if (isClickable && !showCheckbox) {
                Icon(Icons.Default.ChevronRight, contentDescription = "Open Book")
            }
        }
    }
}

@Composable
fun DeletedBooksReport(deletedBooks: List<Book>, onDismiss: () -> Unit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    Icons.Default.Warning, 
                    contentDescription = null, 
                    tint = MaterialTheme.colorScheme.onErrorContainer
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    "Auto-cleanup Report", 
                    style = MaterialTheme.typography.titleMedium, 
                    color = MaterialTheme.colorScheme.onErrorContainer
                )
                Spacer(modifier = Modifier.weight(1f))
                IconButton(onClick = onDismiss) {
                    Icon(Icons.Default.Close, contentDescription = "Dismiss")
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                "The following books were automatically deleted because they were stuck in conversion for more than 2 hours:", 
                style = MaterialTheme.typography.bodyMedium, 
                color = MaterialTheme.colorScheme.onErrorContainer
            )
            Spacer(modifier = Modifier.height(8.dp))
            deletedBooks.forEach { book ->
                Text(
                    "• ${book.title}", 
                    style = MaterialTheme.typography.bodySmall, 
                    color = MaterialTheme.colorScheme.onErrorContainer
                )
            }
        }
    }
}

@Composable
fun SettingsScreen(appPrefs: AppPreferences, onSaved: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var serverUrlInput by remember { mutableStateOf(appPrefs.serverUrl) }
    var isTestingConnection by remember { mutableStateOf(false) }
    var connectionStatus by remember { mutableStateOf<Pair<Boolean, String>?>(null) }
    var showUrlError by remember { mutableStateOf(false) }

    fun isValidUrl(url: String): Boolean {
        return url.isNotBlank() && url.startsWith("http") && (url.contains("playit.gg") || url.contains("cloudflare") || url.contains("trycloudflare.com"))
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        if (appPrefs.isFirstLaunch) {
            Icon(
                Icons.Default.CloudSync,
                contentDescription = null,
                modifier = Modifier
                    .size(80.dp)
                    .align(Alignment.CenterHorizontally),
                tint = MaterialTheme.colorScheme.primary
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Welcome to TTS Reader!",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth()
            )
            Text(
                text = "First, let's connect to your Colab server.",
                style = MaterialTheme.typography.bodyLarge,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth(),
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(16.dp))
        } else {
            Text(
                text = "Server Configuration",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold
            )
        }

        OutlinedTextField(
            value = serverUrlInput,
            onValueChange = {
                serverUrlInput = it.trim()
                connectionStatus = null
                showUrlError = false
            },
            label = { Text("Server URL") },
            placeholder = { Text("https://xxxxx.trycloudflare.com") },
            leadingIcon = { Icon(Icons.Default.Link, contentDescription = null) },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            isError = showUrlError,
            supportingText = {
                if (showUrlError) {
                    Text("Please enter a valid server URL")
                }
            }
        )

        if (!appPrefs.isFirstLaunch) {
            Button(
                onClick = {
                    if (!isValidUrl(serverUrlInput)) {
                        showUrlError = true
                        return@Button
                    }

                    scope.launch {
                        isTestingConnection = true
                        connectionStatus = null

                        val isOnline = withContext(Dispatchers.IO) {
                            ServerAPI.checkServerStatus(serverUrlInput)
                        }

                        connectionStatus = if (isOnline) {
                            Pair(true, "✅ Connection successful!")
                        } else {
                            Pair(false, "❌ Cannot connect. Check URL and ensure Colab server is running.")
                        }

                        isTestingConnection = false
                    }
                },
                modifier = Modifier.fillMaxWidth(),
                enabled = !isTestingConnection && serverUrlInput.isNotBlank()
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    if (isTestingConnection) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(20.dp),
                            strokeWidth = 2.dp,
                            color = LocalContentColor.current
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Testing...")
                    } else {
                        Icon(Icons.Default.Sync, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Test Connection")
                    }
                }
            }
        }

        connectionStatus?.let { (isSuccess, status) ->
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = if (isSuccess)
                        Color(0xFFDCEDC8)
                    else
                        Color(0xFFFFCDD2)
                )
            ) {
                Text(
                    text = status,
                    modifier = Modifier.padding(16.dp),
                    style = MaterialTheme.typography.bodyMedium,
                    color = if (isSuccess) Color(0xFF388E3C) else Color(0xFFD32F2F)
                )
            }
        }

        Spacer(modifier = Modifier.weight(1f))

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.secondaryContainer
            )
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        Icons.Outlined.Info,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onSecondaryContainer
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        "How to get your server URL:",
                        fontWeight = FontWeight.SemiBold,
                        style = MaterialTheme.typography.titleSmall
                    )
                }
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = """1. Run the Python script in Google Colab.
2. Wait for the Cloudflare tunnel URL.
3. Copy the full URL (starting with https://).
4. Paste it into the field above.""",
                    style = MaterialTheme.typography.bodySmall,
                    lineHeight = 18.sp
                )
            }
        }

        Button(
            onClick = {
                if (!isValidUrl(serverUrlInput)) {
                    showUrlError = true
                    return@Button
                }

                if (appPrefs.isFirstLaunch) {
                    scope.launch {
                        isTestingConnection = true
                        connectionStatus = null
                        val isOnline = withContext(Dispatchers.IO) {
                            ServerAPI.checkServerStatus(serverUrlInput)
                        }
                        if (isOnline) {
                            appPrefs.serverUrl = serverUrlInput
                            onSaved()
                        } else {
                            connectionStatus = Pair(false, "❌ Connection failed. Please check the URL.")
                        }
                        isTestingConnection = false
                    }
                } else {
                    appPrefs.serverUrl = serverUrlInput
                    onSaved()
                }
            },
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            enabled = !isTestingConnection
        ) {
            if (isTestingConnection && appPrefs.isFirstLaunch) {
                CircularProgressIndicator(modifier = Modifier.size(28.dp))
            } else {
                Icon(Icons.Default.Save, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text(if (appPrefs.isFirstLaunch) "Save & Continue" else "Save Changes")
            }
        }
    }
}