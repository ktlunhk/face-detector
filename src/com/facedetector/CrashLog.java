package com.facedetector;

import android.app.Application;
import android.content.ContentValues;
import android.content.ContentResolver;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class CrashLog extends Application {
    private static CrashLog instance;
    private static Uri downloadUri;
    private static File fallback;
    private static Thread.UncaughtExceptionHandler previous;

    public void onCreate() {
        super.onCreate();
        instance = this;
        fallback = new File(getFilesDir(), "FaceDetector_crash.txt");
        try {
            if (Build.VERSION.SDK_INT >= 29) {
                ContentValues values = new ContentValues();
                values.put(MediaStore.Downloads.DISPLAY_NAME, "FaceDetector_crash.txt");
                values.put(MediaStore.Downloads.MIME_TYPE, "text/plain");
                values.put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS);
                downloadUri = getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values);
            } else {
                File dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
                if (!dir.exists()) dir.mkdirs();
            }
        } catch (Throwable t) { Log.e("FaceDetector", "Cannot create public log", t); }
        write("Application started. SDK=" + Build.VERSION.SDK_INT + " ABI=" + Build.SUPPORTED_ABIS[0]);
        previous = Thread.getDefaultUncaughtExceptionHandler();
        Thread.setDefaultUncaughtExceptionHandler(new Thread.UncaughtExceptionHandler() {
            public void uncaughtException(Thread thread, Throwable error) {
                write("FATAL thread=" + thread.getName() + "\n" + stack(error));
                if (previous != null) previous.uncaughtException(thread, error);
                else android.os.Process.killProcess(android.os.Process.myPid());
            }
        });
    }

    public static String stack(Throwable t) {
        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);
        t.printStackTrace(pw);
        pw.flush();
        return sw.toString();
    }

    public static synchronized void write(String message) {
        Log.e("FaceDetector", message);
        if (instance == null) return;
        String stamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).format(new Date());
        byte[] bytes;
        try { bytes = (stamp + " " + message + "\n").getBytes("UTF-8"); }
        catch (Exception e) { return; }
        OutputStream out = null;
        try {
            out = new FileOutputStream(fallback, true);
            out.write(bytes);
        } catch (Throwable t) { Log.e("FaceDetector", "Internal log failed", t); }
        finally { if (out != null) try { out.close(); } catch (Exception ignored) {} }
        out = null;
        try {
            if (Build.VERSION.SDK_INT >= 29 && downloadUri != null) {
                out = instance.getContentResolver().openOutputStream(downloadUri, "wa");
            } else if (Build.VERSION.SDK_INT < 29) {
                out = new FileOutputStream(new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "FaceDetector_crash.txt"), true);
            }
            if (out != null) out.write(bytes);
        } catch (Throwable t) { Log.e("FaceDetector", "Download log failed", t); }
        finally { if (out != null) try { out.close(); } catch (Exception ignored) {} }
    }
}
