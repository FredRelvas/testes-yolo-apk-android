// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.util.Log
import java.io.File
import java.util.Locale
import java.nio.ByteBuffer
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Grava MP4 (H.264) a partir de [Bitmap] ARGB ja com deteccoes desenhadas.
 */
internal class AnnotatedVideoRecorder(
    private val outputFile: File,
    private val frameRate: Int = 12,
    private val bitrate: Int = 3_500_000,
    private val iFrameIntervalSec: Int = 2
) {
    companion object {
        private const val TAG = "AnnotatedVideoRecorder"
        private const val MIME = MediaFormat.MIMETYPE_VIDEO_AVC
        private const val MAX_QUEUE = 3
    }

    private data class FrameJob(val bitmap: Bitmap)

    private val queue = ArrayBlockingQueue<FrameJob>(MAX_QUEUE)
    private val stopRequested = AtomicBoolean(false)
    private var worker: Thread? = null

    @Volatile
    var isRecording: Boolean = false
        private set

    fun start() {
        if (isRecording) return
        stopRequested.set(false)
        isRecording = true
        worker = Thread({ encodeLoop() }, "AnnotatedVideoRecorder").apply { start() }
        Log.d(TAG, "Recording -> ${outputFile.absolutePath}")
    }

    fun offerFrame(frame: Bitmap) {
        if (!isRecording || stopRequested.get()) {
            frame.recycle()
            return
        }
        val job = FrameJob(frame)
        if (!queue.offer(job)) {
            queue.poll()?.bitmap?.recycle()
            queue.offer(job)
        }
    }

    fun stop(onComplete: (String?) -> Unit) {
        if (!isRecording) {
            onComplete(null)
            return
        }
        stopRequested.set(true)
        worker?.join(60_000)
        worker = null
        isRecording = false
        while (queue.isNotEmpty()) {
            queue.poll()?.bitmap?.recycle()
        }
        val path =
            if (outputFile.exists() && outputFile.length() > 64) outputFile.absolutePath else null
        onComplete(path)
    }

    private fun pickColorFormat(caps: MediaCodecInfo.CodecCapabilities): Int {
        val prefs = intArrayOf(
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar
        )
        for (p in prefs) {
            for (f in caps.colorFormats) {
                if (f == p) return f
            }
        }
        return caps.colorFormats[0]
    }

    private fun encodeLoop() {
        var muxer: MediaMuxer? = null
        var codec: MediaCodec? = null
        var trackIndex = -1
        var muxerStarted = false
        var presentationTimeUs = 0L
        val frameIntervalUs = 1_000_000L / frameRate
        var encWidth = 0
        var encHeight = 0
        var colorFormat = 0

        try {
            val first = waitFirstFrame() ?: return
            var bmp = first.bitmap
            encWidth = bmp.width and 0x7FFFFFFE
            encHeight = bmp.height and 0x7FFFFFFE
            if (encWidth < 32 || encHeight < 32) {
                bmp.recycle()
                return
            }
            val maxLong = 960
            val longE = if (encWidth >= encHeight) encWidth.toFloat() else encHeight.toFloat()
            if (longE > maxLong) {
                val scale = maxLong / longE
                encWidth = (encWidth * scale).toInt() and 0x7FFFFFFE
                encHeight = (encHeight * scale).toInt() and 0x7FFFFFFE
                val scaled = Bitmap.createScaledBitmap(bmp, encWidth, encHeight, true)
                bmp.recycle()
                bmp = scaled
            }

            val c = MediaCodec.createEncoderByType(MIME)
            colorFormat = pickColorFormat(c.codecInfo.getCapabilitiesForType(MIME))
            val format = MediaFormat.createVideoFormat(MIME, encWidth, encHeight).apply {
                setInteger(MediaFormat.KEY_COLOR_FORMAT, colorFormat)
                setInteger(MediaFormat.KEY_BIT_RATE, bitrate)
                setInteger(MediaFormat.KEY_FRAME_RATE, frameRate)
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, iFrameIntervalSec)
            }
            c.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            c.start()
            codec = c
            muxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)

            var inputDone = false
            val bufferInfo = MediaCodec.BufferInfo()

            fun scaleToEncode(src: Bitmap): Bitmap {
                if (src.width == encWidth && src.height == encHeight) {
                    val cpy = src.copy(Bitmap.Config.ARGB_8888, true)
                    src.recycle()
                    return cpy
                }
                val out = Bitmap.createScaledBitmap(src, encWidth, encHeight, true)
                if (out != src) src.recycle()
                return out
            }

            fun submitFrame(b: Bitmap) {
                val frame = scaleToEncode(b)
                val inIdx = c.dequeueInputBuffer(30_000)
                if (inIdx < 0) {
                    frame.recycle()
                    return
                }
                val buf = c.getInputBuffer(inIdx) ?: run {
                    frame.recycle()
                    return
                }
                buf.clear()
                val sizeNeeded = encWidth * encHeight * 3 / 2
                if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar) {
                    bitmapToI420(frame, encWidth, encHeight, buf)
                } else {
                    bitmapToNv12(frame, encWidth, encHeight, buf)
                }
                c.queueInputBuffer(inIdx, 0, sizeNeeded, presentationTimeUs, 0)
                presentationTimeUs += frameIntervalUs
                frame.recycle()
            }

            fun submitEos() {
                val inIdx = c.dequeueInputBuffer(30_000)
                if (inIdx >= 0) {
                    c.queueInputBuffer(
                        inIdx,
                        0,
                        0,
                        presentationTimeUs,
                        MediaCodec.BUFFER_FLAG_END_OF_STREAM
                    )
                }
            }

            fun drainOnce(): Boolean {
                var sawOutputEos = false
                val timeoutUs = if (inputDone) 20_000L else 0L
                val mx = muxer
                if (mx == null) return sawOutputEos
                while (true) {
                    val outIdx = c.dequeueOutputBuffer(bufferInfo, timeoutUs)
                    when {
                        outIdx == MediaCodec.INFO_TRY_AGAIN_LATER -> return sawOutputEos
                        outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                            if (!muxerStarted) {
                                trackIndex = mx.addTrack(c.outputFormat)
                                mx.start()
                                muxerStarted = true
                            }
                        }
                        outIdx >= 0 -> {
                            val encoded = c.getOutputBuffer(outIdx)
                            if (muxerStarted && encoded != null && bufferInfo.size > 0) {
                                encoded.position(bufferInfo.offset)
                                encoded.limit(bufferInfo.offset + bufferInfo.size)
                                mx.writeSampleData(trackIndex, encoded, bufferInfo)
                            }
                            c.releaseOutputBuffer(outIdx, false)
                            if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                                sawOutputEos = true
                            }
                        }
                        else -> {
                            if (outIdx < 0) {
                                Log.w(TAG, "dequeueOutputBuffer: $outIdx")
                            }
                            return sawOutputEos
                        }
                    }
                }
            }

            submitFrame(bmp)
            bmp.recycle()

            while (true) {
                if (!inputDone) {
                    val job = queue.poll(80, TimeUnit.MILLISECONDS)
                    if (job != null) {
                        submitFrame(job.bitmap)
                    } else if (stopRequested.get()) {
                        submitEos()
                        inputDone = true
                    }
                } else {
                    Thread.sleep(5)
                }
                val outEos = drainOnce()
                if (inputDone && outEos) break
            }

            c.stop()
            c.release()
            codec = null
            if (muxerStarted) {
                muxer?.stop()
            }
            muxer?.release()
            muxer = null
        } catch (e: Exception) {
            Log.e(TAG, "encodeLoop", e)
            try {
                codec?.stop()
            } catch (_: Exception) {
            }
            codec?.release()
            try {
                if (muxerStarted) muxer?.stop()
            } catch (_: Exception) {
            }
            muxer?.release()
        } finally {
            while (queue.isNotEmpty()) {
                queue.poll()?.bitmap?.recycle()
            }
        }
    }

    private fun waitFirstFrame(): FrameJob? {
        while (!stopRequested.get()) {
            val j = queue.poll(300, TimeUnit.MILLISECONDS) ?: continue
            return j
        }
        return null
    }

    private fun bitmapToI420(bitmap: Bitmap, width: Int, height: Int, out: ByteBuffer) {
        val argb = IntArray(width * height)
        bitmap.getPixels(argb, 0, width, 0, 0, width, height)
        val ySize = width * height
        val uvW = width / 2
        val uvH = height / 2
        val uvSize = uvW * uvH
        val y = ByteArray(ySize)
        val u = ByteArray(uvSize)
        val v = ByteArray(uvSize)
        for (j in 0 until height) {
            for (i in 0 until width) {
                val c = argb[j * width + i]
                val r = (c shr 16) and 0xFF
                val g = (c shr 8) and 0xFF
                val b = c and 0xFF
                val yv = ((66 * r + 129 * g + 25 * b + 128) shr 8) + 16
                y[j * width + i] = yv.coerceIn(0, 255).toByte()
            }
        }
        var uv = 0
        for (j in 0 until height step 2) {
            for (i in 0 until width step 2) {
                val c = argb[j * width + i]
                val r = (c shr 16) and 0xFF
                val g = (c shr 8) and 0xFF
                val b = c and 0xFF
                val uu = ((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128
                val vv = ((112 * r - 94 * g - 18 * b + 128) shr 8) + 128
                u[uv] = uu.coerceIn(0, 255).toByte()
                v[uv] = vv.coerceIn(0, 255).toByte()
                uv++
            }
        }
        out.put(y)
        out.put(u)
        out.put(v)
    }

    private fun bitmapToNv12(bitmap: Bitmap, width: Int, height: Int, out: ByteBuffer) {
        val argb = IntArray(width * height)
        bitmap.getPixels(argb, 0, width, 0, 0, width, height)
        val ySize = width * height
        val uvCount = width * height / 2
        val y = ByteArray(ySize)
        val uv = ByteArray(uvCount)
        for (j in 0 until height) {
            for (i in 0 until width) {
                val c = argb[j * width + i]
                val r = (c shr 16) and 0xFF
                val g = (c shr 8) and 0xFF
                val b = c and 0xFF
                val yv = ((66 * r + 129 * g + 25 * b + 128) shr 8) + 16
                y[j * width + i] = yv.coerceIn(0, 255).toByte()
            }
        }
        var k = 0
        for (j in 0 until height step 2) {
            for (i in 0 until width step 2) {
                val c = argb[j * width + i]
                val r = (c shr 16) and 0xFF
                val g = (c shr 8) and 0xFF
                val b = c and 0xFF
                val uu = ((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128
                val vv = ((112 * r - 94 * g - 18 * b + 128) shr 8) + 128
                uv[k++] = uu.coerceIn(0, 255).toByte()
                uv[k++] = vv.coerceIn(0, 255).toByte()
            }
        }
        out.put(y)
        out.put(uv)
    }
}

/** Mesma rotacao que [ObjectDetector.predict] para camera em portrait. */
internal fun rotateCameraBitmapForDetector(
    bitmap: Bitmap,
    isLandscape: Boolean,
    isFrontCamera: Boolean
): Bitmap {
    if (isLandscape) return bitmap.copy(Bitmap.Config.ARGB_8888, true)
    val k = if (isFrontCamera) 1 else 3
    val kk = ((k % 4) + 4) % 4
    if (kk == 0) return bitmap.copy(Bitmap.Config.ARGB_8888, true)
    val m = android.graphics.Matrix()
    m.postRotate(-90f * kk)
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
}

/** Mesma convencao que Flutter [EpiOverlay] / `boxKeyForRecording`. */
internal fun epiBoxKey(cls: String, xywhn: RectF): String =
    "${cls}_${String.format(Locale.US, "%.4f", xywhn.left)}_${String.format(Locale.US, "%.4f", xywhn.top)}"

/** Caixas verde/vermelho alinhadas a overlay EPI; [redKeys] vindo do Flutter. */
internal fun drawEpiStyleBoxesOnBitmap(
    bitmap: Bitmap,
    result: YOLOResult,
    task: YOLOTask,
    flipHorizontal: Boolean,
    redKeys: Set<String>
): Bitmap {
    val out = bitmap.copy(Bitmap.Config.ARGB_8888, true)
    if (task != YOLOTask.DETECT || result.boxes.isEmpty()) return out

    val canvas = Canvas(out)
    val w = out.width.toFloat()
    val h = out.height.toFloat()
    val strokeW = kotlin.math.max(3f, kotlin.math.max(w, h) / 400f)
    val textSize = kotlin.math.max(22f, kotlin.math.max(w, h) / 55f)
    val corner = strokeW * 1.5f
    val pad = kotlin.math.max(4f, strokeW)

    val greenStroke = Color.parseColor("#FF4CAF50")
    val redStroke = Color.parseColor("#FFE53935")
    val greenFill = Color.parseColor("#334CAF50")
    val redFill = Color.parseColor("#66E53935")

    val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = strokeW
    }
    val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        this.textSize = textSize
        isFakeBoldText = true
    }
    val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }

    for (box in result.boxes) {
        var left = box.xywh.left
        var top = box.xywh.top
        var right = box.xywh.right
        var bottom = box.xywh.bottom
        if (flipHorizontal) {
            val nl = w - right
            val nr = w - left
            left = nl
            right = nr
        }
        val isRed = redKeys.contains(epiBoxKey(box.cls, box.xywhn))
        val stroke = if (isRed) redStroke else greenStroke
        val fill = if (isRed) redFill else greenFill

        fillPaint.color = fill
        strokePaint.color = stroke
        canvas.drawRoundRect(RectF(left, top, right, bottom), corner, corner, fillPaint)
        canvas.drawRoundRect(RectF(left, top, right, bottom), corner, corner, strokePaint)

        val label = "${box.cls} ${"%.0f".format(box.conf * 100)}%"
        val tw = textPaint.measureText(label)
        val fm = textPaint.fontMetrics
        val labelH = fm.descent - fm.ascent + pad * 2f
        val labelTop = (top - labelH).coerceAtLeast(0f)
        val labelRect = RectF(left, labelTop, left + tw + pad * 2f, labelTop + labelH)
        labelBgPaint.color = stroke
        canvas.drawRoundRect(labelRect, corner, corner, labelBgPaint)
        val baseline = labelTop + pad - fm.ascent
        canvas.drawText(label, left + pad, baseline, textPaint)
    }
    return out
}
