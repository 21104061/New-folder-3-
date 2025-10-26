package com.hollow.ttsreader

import androidx.media3.common.C
import androidx.media3.datasource.DataSource
import androidx.media3.datasource.DataSpec
import androidx.media3.datasource.TransferListener
import java.io.IOException
import java.io.InputStream

class InputStreamDataSource(private val inputStream: InputStream) : DataSource {
    private var opened = false

    override fun addTransferListener(transferListener: TransferListener) {
        // Do nothing
    }

    override fun open(dataSpec: DataSpec): Long {
        opened = true
        return C.LENGTH_UNSET.toLong() // unknown length
    }

    override fun read(buffer: ByteArray, offset: Int, readLength: Int): Int {
        if (!opened) throw IOException("DataSource not opened")
        return inputStream.read(buffer, offset, readLength)
    }

    override fun getUri() = null

    override fun close() {
        if (opened) {
            opened = false
            try { inputStream.close() } catch (_: IOException) {}
        }
    }

    override fun getResponseHeaders(): MutableMap<String, MutableList<String>> {
        return mutableMapOf()
    }
}
