// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo_example

import android.content.Context
import android.os.BatteryManager
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import kotlin.math.abs

class MainActivity : FlutterActivity() {

    companion object {
        private const val CHANNEL = "poc_epi/power"
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "currentNowMicroAmps" -> result.success(readCurrentNow())
                    else -> result.notImplemented()
                }
            }
    }

    /**
     * Le BatteryManager.BATTERY_PROPERTY_CURRENT_NOW em microamperes.
     *
     * Convencao oficial: positivo = carregando, negativo = descarregando.
     * Mas alguns fabricantes (Samsung antigos) invertem o sinal -- por
     * isso retornamos o valor cru e a camada Dart aplica abs() para
     * exibir a magnitude do consumo.
     *
     * Retorna `null` se o dispositivo nao suportar a propriedade.
     */
    private fun readCurrentNow(): Long? {
        return try {
            val bm = getSystemService(Context.BATTERY_SERVICE) as BatteryManager
            val value = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
            // Alguns devices retornam Long.MIN_VALUE quando nao suportado.
            if (value == Long.MIN_VALUE || value == 0L) null
            else abs(value)
        } catch (e: Exception) {
            null
        }
    }
}
