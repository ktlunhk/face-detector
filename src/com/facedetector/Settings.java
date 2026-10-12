package com.facedetector;

import android.content.SharedPreferences;

/** Persistent detection configuration, using existing preference keys. */
public final class Settings {
    private Settings() {}
    public static void save(SharedPreferences prefs, float scaleFactor, int minNeighbors,
            int minFace, int interval, int holdMs, float smoothing,
            float yuNetConfidence, float yuNetNms, int yuNetTopK) {
        prefs.edit().putFloat("scale",scaleFactor).putInt("neighbors",minNeighbors)
            .putInt("face",minFace).putInt("interval",interval).putInt("hold",holdMs)
            .putFloat("smooth",smoothing).putFloat("yunet_conf",yuNetConfidence)
            .putFloat("yunet_nms",yuNetNms).putInt("yunet_topk",yuNetTopK).apply();
    }
}
