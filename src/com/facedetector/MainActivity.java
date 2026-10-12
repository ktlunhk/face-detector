package com.facedetector;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.widget.ImageView;
import java.io.IOException;
import android.Manifest;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.os.Bundle;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.TextView;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.Button;
import android.view.Gravity;
import android.content.SharedPreferences;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.Drawable;
import android.content.res.ColorStateList;
import android.content.Context;
import android.graphics.RectF;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.CvType;
import org.opencv.core.Core;
import org.opencv.core.Size;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.FaceDetectorYN;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.List;

@SuppressWarnings("deprecation")
public class MainActivity extends Activity implements SurfaceHolder.Callback, Camera.PreviewCallback {
    private static final int PICK_IMAGE = 202;
    private final boolean useYuNet = true;
    private FaceDetectorYN yuNet;
    private Net faceMeshNet;
    private volatile String meshDiagnostic = "OFF";
    private final Object meshLock = new Object();
    private static final int[] MESH_EDGES = new int[] {
        0,11,0,37,0,164,0,167,0,267,0,302,0,312,1,4,1,19,1,44,1,125,1,274,1,354,2,94,2,141,2,164,
        2,370,3,5,3,51,3,174,3,195,3,196,3,236,4,5,4,44,4,45,4,51,4,274,4,275,4,281,5,51,5,195,
        5,248,5,281,6,122,6,168,6,193,6,197,6,351,6,417,7,25,7,110,7,161,7,163,7,246,8,9,8,55,8,168,
        8,193,8,285,8,417,9,55,9,107,9,151,9,285,9,336,10,109,10,151,10,338,11,12,11,37,11,72,11,82,11,312,
        12,13,12,82,12,312,13,14,13,82,13,312,13,317,14,15,14,82,14,87,14,317,15,16,15,86,15,87,15,316,15,317,
        16,17,16,84,16,85,16,86,16,314,16,315,16,316,17,18,17,83,17,84,17,313,17,314,18,83,18,200,18,313,18,421,
        19,94,19,125,19,354,19,370,20,97,20,99,20,238,20,239,20,241,20,242,21,54,21,68,21,71,21,162,22,23,22,26,
        22,145,22,153,22,154,22,231,22,232,23,24,23,144,23,145,23,230,23,231,24,110,24,144,24,163,24,229,24,230,25,31,
        25,33,25,110,25,130,25,228,25,246,26,112,26,154,26,155,26,232,27,28,27,29,27,159,27,160,27,222,27,223,28,56,
        28,158,28,159,28,222,29,30,29,160,29,161,29,223,29,224,30,161,30,224,30,225,30,246,30,247,31,35,31,111,31,117,
        31,130,31,226,31,228,32,140,32,170,32,194,32,201,32,208,32,211,33,130,33,246,33,247,34,127,34,139,34,143,34,156,
        34,234,35,111,35,124,35,143,35,226,36,101,36,129,36,142,36,203,36,205,36,206,37,39,37,72,37,73,37,81,37,167,
        38,72,38,81,38,82,39,40,39,73,39,74,39,80,39,165,39,167,40,74,40,92,40,165,40,183,40,184,40,185,40,191,
        41,73,41,80,41,81,41,88,41,178,42,74,42,80,42,88,42,95,42,191,43,57,43,61,43,106,43,146,43,202,43,204,
        44,45,44,125,44,237,44,238,44,241,45,51,45,134,45,220,45,237,46,53,46,63,46,70,46,124,46,156,46,225,47,100,
        47,114,47,120,47,121,47,126,47,217,48,49,48,64,48,102,48,115,48,219,49,102,49,115,49,129,49,131,49,142,49,209,
        50,101,50,117,50,118,50,123,50,147,50,187,50,205,51,134,51,236,52,53,52,65,52,66,52,105,52,223,52,224,53,63,
        53,105,53,224,53,225,54,68,54,103,54,104,55,107,55,189,55,193,55,221,56,157,56,158,56,190,56,221,56,222,57,61,
        57,186,57,202,57,212,57,216,58,138,58,172,58,192,58,215,59,75,59,166,59,235,59,240,60,75,60,79,60,99,60,239,
        61,76,61,146,61,185,61,186,62,76,62,77,62,78,62,146,62,184,63,68,63,70,63,71,63,104,63,105,64,98,64,102,
        64,129,64,203,64,219,64,235,65,66,65,107,65,221,65,222,65,223,66,69,66,105,66,107,66,108,67,69,67,103,67,104,
        67,109,68,71,68,104,69,104,69,105,69,108,69,109,70,71,70,139,70,156,71,139,71,162,72,81,72,82,73,80,73,81,
        74,80,74,191,75,79,75,99,75,166,75,167,75,218,75,240,76,146,76,184,76,185,77,78,77,91,77,96,77,146,78,96,
        78,183,78,184,79,218,79,220,79,237,79,239,80,88,81,82,81,178,82,87,82,178,83,84,83,182,83,194,83,200,83,201,
        84,85,84,181,84,182,85,86,85,179,85,180,85,181,86,87,86,178,86,179,87,178,88,89,88,95,88,178,88,179,88,180,
        89,90,89,95,89,180,90,91,90,95,90,96,90,180,90,181,91,96,91,106,91,146,91,181,91,182,92,165,92,185,92,186,
        92,206,92,216,93,116,93,123,93,137,93,227,93,234,94,125,94,141,94,370,95,96,95,183,95,191,96,183,97,99,97,164,
        97,167,97,242,98,165,98,167,98,203,98,235,98,240,99,167,99,239,100,101,100,119,100,120,100,126,100,142,101,118,101,119,
        101,142,101,205,102,129,103,104,104,105,106,146,106,182,106,204,107,108,107,221,108,109,108,151,110,163,110,228,110,229,111,116,
        111,117,111,143,112,133,112,155,112,232,112,233,112,243,113,124,113,225,113,226,113,247,114,121,114,128,114,174,114,188,114,217,
        115,131,115,166,115,218,115,219,115,220,116,117,116,123,116,143,116,227,116,234,117,118,117,123,117,228,118,119,118,228,118,229,
        119,120,119,229,119,230,120,121,120,230,120,231,121,128,121,231,121,232,122,188,122,193,122,196,122,197,122,245,123,137,123,147,
        124,143,124,156,124,225,124,226,125,141,125,241,125,242,126,142,126,198,126,209,126,217,127,139,127,162,127,234,128,188,128,232,
        128,233,128,245,129,142,129,203,130,226,130,247,131,134,131,198,131,209,131,220,132,147,132,177,132,213,132,215,133,155,133,173,
        133,190,133,243,134,198,134,220,134,236,135,136,135,172,135,210,135,214,136,150,136,169,136,172,136,210,137,147,137,177,138,172,
        138,192,138,214,139,156,139,162,140,149,140,170,140,171,140,176,140,208,141,164,141,242,142,209,143,156,143,234,144,145,144,159,
        144,160,144,163,145,153,145,158,145,159,147,177,147,187,147,213,148,152,148,171,148,175,148,176,149,150,149,170,149,176,150,169,
        150,170,151,336,151,337,152,175,152,377,153,154,153,157,153,158,154,155,154,157,154,173,155,173,157,158,157,173,157,190,158,159,
        159,160,160,161,160,163,161,163,161,246,164,167,164,242,164,267,164,370,164,393,165,167,165,203,165,206,166,218,166,219,166,235,
        167,240,168,193,168,417,169,170,169,210,169,211,170,211,171,175,171,176,171,199,171,208,172,214,173,190,174,188,174,196,174,217,
        174,236,175,199,175,377,175,396,177,215,178,179,179,180,180,181,181,182,182,194,182,204,183,184,183,191,184,185,185,186,186,216,
        187,192,187,205,187,207,187,213,188,196,188,245,189,190,189,193,189,221,189,243,189,244,190,221,190,243,192,207,192,213,192,214,
        192,215,193,244,193,245,194,201,194,204,194,211,195,196,195,197,195,248,195,419,196,197,197,351,197,419,198,209,198,217,198,236,
        199,200,199,208,199,396,199,428,200,201,200,208,200,421,200,428,201,208,202,204,202,210,202,212,203,206,204,210,204,211,205,206,
        205,207,205,216,206,216,207,214,207,216,210,211,210,212,210,214,212,214,212,216,213,215,214,216,217,236,218,220,219,235,220,237,
        221,222,222,223,223,224,224,225,225,247,226,247,227,234,228,229,229,230,230,231,231,232,232,233,233,243,233,244,233,245,235,240,
        237,238,237,239,238,239,238,241,241,242,243,244,244,245,246,247,248,281,248,399,248,419,248,456,249,255,249,339,249,388,249,390,
        249,466,250,326,250,328,250,458,250,461,250,462,251,284,251,298,251,301,251,389,252,253,252,256,252,380,252,381,252,451,252,452,
        253,254,253,374,253,380,253,450,253,451,254,339,254,373,254,374,254,390,254,449,254,450,255,261,255,263,255,339,255,359,255,448,
        255,466,256,341,256,381,256,382,256,452,257,258,257,259,257,386,257,387,257,442,257,443,258,286,258,385,258,386,258,442,259,260,
        259,387,259,388,259,443,259,444,260,342,260,388,260,444,260,445,260,466,260,467,261,265,261,340,261,346,261,359,261,446,261,448,
        262,369,262,395,262,418,262,421,262,428,262,431,263,359,263,466,263,467,264,356,264,372,264,447,264,454,265,340,265,342,265,353,
        265,372,265,446,266,330,266,358,266,371,266,423,266,425,266,426,267,269,267,302,267,303,267,311,267,393,268,302,268,311,268,312,
        268,402,269,270,269,303,269,304,269,310,269,391,269,393,270,304,270,322,270,391,270,407,270,408,270,409,271,303,271,310,271,311,
        271,318,271,402,272,304,272,310,272,318,272,324,272,415,273,287,273,291,273,335,273,375,273,422,273,424,274,275,274,354,274,457,
        274,458,274,459,274,461,275,281,275,363,275,440,275,457,276,283,276,300,276,353,276,383,276,445,277,329,277,343,277,349,277,350,
        277,355,277,437,278,279,278,294,278,331,278,344,278,439,279,331,279,344,279,358,279,360,279,371,279,429,280,330,280,346,280,347,
        280,352,280,376,280,411,280,425,281,363,281,456,282,283,282,295,282,296,282,334,282,443,282,444,283,293,283,300,283,334,283,444,
        283,445,284,298,284,332,284,333,285,336,285,413,285,417,285,441,285,442,286,384,286,385,286,414,286,441,286,442,287,291,287,409,
        287,410,287,422,287,432,288,367,288,416,288,433,288,435,289,305,289,309,289,392,289,438,289,460,290,305,290,328,290,458,290,459,
        291,306,291,375,291,409,292,306,292,307,292,308,292,375,292,408,293,298,293,300,293,301,293,333,293,334,294,327,294,331,294,358,
        294,423,294,439,294,455,295,296,295,336,295,442,295,443,296,299,296,334,296,336,296,337,297,299,297,332,297,333,297,338,298,301,
        298,333,299,333,299,334,299,337,299,338,300,301,300,368,300,383,300,389,301,389,302,311,302,312,303,310,303,311,304,310,304,407,
        304,415,305,309,305,328,305,459,305,460,306,375,306,408,306,409,307,308,307,321,307,325,307,375,308,325,308,407,308,408,309,438,
        309,440,309,457,309,459,310,318,311,402,312,317,312,402,313,314,313,406,313,418,313,421,314,315,314,405,314,406,315,316,315,403,
        315,404,315,405,316,317,316,402,316,403,317,402,318,319,318,324,318,402,318,403,318,404,319,320,319,324,319,404,320,321,320,324,
        320,325,320,404,320,405,321,325,321,335,321,375,321,405,322,391,322,409,322,410,322,426,322,436,323,345,323,352,323,366,323,447,
        324,325,324,415,325,407,325,415,326,328,326,393,326,462,327,391,327,423,327,455,327,460,328,393,328,458,328,460,329,330,329,348,
        329,349,329,355,329,371,330,347,330,348,330,371,330,425,331,358,332,333,333,334,335,375,335,405,335,406,335,424,336,337,336,442,
        337,338,339,390,339,448,339,449,340,345,340,346,340,372,340,454,341,362,341,382,341,452,341,453,341,463,342,353,342,445,342,446,
        342,467,343,350,343,357,343,399,343,412,343,437,344,360,344,438,344,439,344,440,345,346,345,352,345,447,345,454,346,347,346,352,
        346,448,347,348,347,448,347,449,348,349,348,449,348,450,349,350,349,450,349,451,350,357,350,451,350,452,351,412,351,417,351,419,
        351,465,352,366,352,376,353,372,353,383,353,445,354,370,354,461,354,462,355,371,355,420,355,429,355,437,356,368,356,372,356,383,
        356,389,356,447,357,412,357,452,357,453,357,465,358,371,358,423,359,446,359,467,360,363,360,420,360,429,360,440,361,376,361,401,
        361,411,361,433,362,382,362,398,362,414,362,463,363,420,363,440,363,456,364,365,364,397,364,430,364,432,364,434,365,379,365,394,
        365,397,365,430,366,376,366,401,366,447,367,397,367,416,367,434,367,435,368,383,368,389,369,378,369,395,369,396,369,400,369,428,
        370,393,370,462,371,429,372,383,372,454,373,374,373,386,373,387,373,390,374,380,374,385,374,386,376,401,376,411,377,396,377,400,
        378,379,378,395,378,400,379,394,379,395,380,381,380,384,380,385,381,382,381,384,381,398,382,398,384,385,384,398,384,414,385,386,
        386,387,387,388,387,390,388,390,388,466,391,393,391,423,391,426,391,460,392,438,392,439,392,455,392,460,393,460,393,462,394,395,
        394,430,394,431,395,431,396,400,396,428,397,434,397,435,398,414,399,412,399,419,399,437,399,456,401,433,401,435,402,403,403,404,
        404,405,405,406,406,418,406,424,407,408,407,415,408,409,409,410,410,432,410,436,411,425,411,427,411,433,412,419,412,465,413,414,
        413,417,413,441,413,463,413,464,414,441,414,463,416,427,416,433,416,434,417,464,417,465,418,421,418,424,418,431,420,429,420,437,
        420,456,421,428,422,424,422,430,422,432,423,426,424,430,424,431,425,426,425,427,426,427,426,436,427,433,427,434,427,436,430,431,
        430,432,432,434,432,436,433,435,434,436,437,456,438,439,438,440,439,455,440,457,441,442,442,443,443,444,444,445,446,467,447,454,
        448,449,449,450,450,451,451,452,452,453,453,463,453,464,453,465,455,460,457,459,458,459,458,461,461,462,463,464,464,465,466,467,
    };

    private volatile String yuNetDiagnostics = "YuNet not yet run";
    private volatile float yuNetConfidence = 0.70f;
    private volatile float yuNetNms = 0.30f;
    private volatile int yuNetTopK = 500;
    private final Object yuNetLock = new Object();
    private Button engineButton;
    private boolean imageMode = false;
    private ImageFaceView imageView;
    private Bitmap selectedBitmap;
    private Button loadButton, cameraButton, meshButton;
    private volatile boolean meshEnabled = false;
    private int imageGeneration = 0;
    private Camera camera;
    private SurfaceView surface;
    private Overlay overlay;
    private TextView status;
    private CascadeClassifier classifier;
    private int cameraId = -1;
    private int width = 640, height = 480, rotation = 90;
    private boolean busy = false;
    private long lastScan = 0;
    private boolean opencvReady = false;
    private LinearLayout root;
    private FrameLayout previewArea;
    private LinearLayout settingsPanel;
    private float scaleFactor = 1.10f, smoothing = 0.50f;
    private int minNeighbors = 5, minFace = 70, interval = 150, holdMs = 400;
    private RectF[] displayed = new RectF[0];
    private long lastDetected = 0;
    private SharedPreferences prefs;
    private int frameRotation = 90;
    private boolean frontFacing = false;
    private void saveSettings() {
        prefs.edit().putFloat("scale",scaleFactor).putInt("neighbors",minNeighbors)
            .putInt("face",minFace).putInt("interval",interval).putInt("hold",holdMs)
            .putFloat("smooth",smoothing).putFloat("yunet_conf",yuNetConfidence)
            .putFloat("yunet_nms",yuNetNms).putInt("yunet_topk",yuNetTopK).apply();
    }
    private GradientDrawable rounded(int color, int stroke, int radius) {
        GradientDrawable bg = new GradientDrawable();
        bg.setColor(color);
        bg.setCornerRadius(dp(radius));
        if (stroke != 0) bg.setStroke(dp(1), stroke);
        return bg;
    }
    // Lightweight vector icons: no emoji fonts, AndroidX, or external resources.
    private Drawable icon(final int type, final int color) {
        Drawable d = new Drawable() {
            private final Paint p = new Paint(3);
            public void draw(Canvas canvas) {
                android.graphics.Rect bounds = getBounds();
                canvas.save();
                canvas.translate(bounds.left, bounds.top);
                canvas.scale(bounds.width() / 24f, bounds.height() / 24f);
                p.setColor(color); p.setStrokeWidth(1.9f); p.setStrokeCap(Paint.Cap.ROUND);
                p.setStrokeJoin(Paint.Join.ROUND); p.setStyle(Paint.Style.STROKE);
                if (type == 0) { // gallery
                    canvas.drawRoundRect(2, 4, 22, 20, 2, 2, p);
                    canvas.drawCircle(8, 9, 2, p);
                    canvas.drawLine(4, 18, 10, 12, p); canvas.drawLine(10, 12, 14, 16, p);
                    canvas.drawLine(14, 16, 17, 13, p); canvas.drawLine(17, 13, 21, 18, p);
                } else if (type == 1) { // camera
                    canvas.drawRoundRect(2, 7, 22, 20, 3, 3, p);
                    canvas.drawRect(8, 4, 15, 7, p); canvas.drawCircle(12, 13, 4, p);
                } else if (type == 2) { // switch camera
                    canvas.drawArc(3, 3, 21, 21, 210, 230, false, p);
                    canvas.drawArc(3, 3, 21, 21, 30, 145, false, p);
                    canvas.drawLine(4, 10, 3, 4, p); canvas.drawLine(3, 4, 9, 5, p);
                    canvas.drawLine(20, 14, 21, 20, p); canvas.drawLine(21, 20, 15, 19, p);
                } else if (type == 3) { // mesh
                    canvas.drawOval(5, 2, 19, 22, p);
                    canvas.drawLine(12, 2, 12, 22, p); canvas.drawLine(5, 10, 19, 10, p);
                    canvas.drawLine(7, 5, 17, 18, p); canvas.drawLine(17, 5, 7, 18, p);
                } else if (type == 4) { // shield / confidence
                    android.graphics.Path path = new android.graphics.Path();
                    path.moveTo(12,2); path.lineTo(20,5); path.lineTo(19,14);
                    path.quadTo(17,19,12,22); path.quadTo(7,19,5,14);
                    path.lineTo(4,5); path.close(); canvas.drawPath(path,p);
                    canvas.drawLine(9,12,11,14,p); canvas.drawLine(11,14,16,9,p);
                } else if (type == 5) { // overlapping circles / NMS
                    canvas.drawCircle(9,12,6,p); canvas.drawCircle(15,12,6,p);
                } else if (type == 6) { // list / max detections
                    for(int y=6;y<=18;y+=6){canvas.drawCircle(4,y,1,p);canvas.drawLine(9,y,21,y,p);}
                } else if (type == 7) { // clock
                    canvas.drawCircle(12,12,9,p); canvas.drawLine(12,12,12,6,p);
                    canvas.drawLine(12,12,16,14,p);
                } else if (type == 8) { // stopwatch
                    canvas.drawCircle(12,13,8,p);canvas.drawLine(12,13,12,8,p);
                    canvas.drawLine(9,2,15,2,p);canvas.drawLine(12,2,12,5,p);
                } else if (type == 9) { // smoothing waves
                    for(int y=7;y<=17;y+=5){
                        android.graphics.Path path = new android.graphics.Path();
                        path.moveTo(2,y);path.cubicTo(6,y-4,8,y+4,12,y);
                        path.cubicTo(16,y-4,18,y+4,22,y);canvas.drawPath(path,p);
                    }
                } else { // settings gear
                    canvas.drawCircle(12,12,7,p);canvas.drawCircle(12,12,3,p);
                    for(int i=0;i<8;i++){double a=i*Math.PI/4;
                        canvas.drawLine((float)(12+8*Math.cos(a)),(float)(12+8*Math.sin(a)),
                            (float)(12+11*Math.cos(a)),(float)(12+11*Math.sin(a)),p);}
                }
                canvas.restore();
            }
            public void setAlpha(int alpha) { p.setAlpha(alpha); }
            public void setColorFilter(android.graphics.ColorFilter filter) { p.setColorFilter(filter); }
            public int getOpacity() { return android.graphics.PixelFormat.TRANSLUCENT; }
            public int getIntrinsicWidth() { return dp(20); }
            public int getIntrinsicHeight() { return dp(20); }
        };
        d.setBounds(0, 0, dp(20), dp(20));
        return d;
    }
    private void setIcon(TextView view, int type, int color) {
        view.setCompoundDrawables(icon(type, color), null, null, null);
        view.setCompoundDrawablePadding(dp(8));
    }
    private void styleAction(Button button, boolean selected) {
        button.setAllCaps(false);
        button.setTextSize(12);
        button.setTextColor(selected ? 0xFF06131E : 0xFFE8F2FF);
        button.setBackground(rounded(selected ? 0xFF22D3EE : 0xFF1C2C41,
            selected ? 0xFF22D3EE : 0xFF314A66, 16));
        button.setPadding(dp(4), dp(5), dp(4), dp(5));
        button.setMinHeight(dp(40));
        button.setGravity(Gravity.CENTER);
    }
    private LinearLayout.LayoutParams buttonParams() {
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, dp(42), 1f);
        lp.setMargins(dp(4), dp(3), dp(4), dp(3));
        return lp;
    }
    private void makeSlider(LinearLayout panel, final String name, int initial, final int min, final int step, int max, final int kind) {
        final TextView label = new TextView(this);
        label.setTextColor(0xFFE2EAF4); label.setPadding(dp(5),dp(3),dp(5),0);
        setIcon(label, kind == 6 ? 4 : kind == 7 ? 5 : kind == 8 ? 6 : kind == 3 ? 7 : kind == 4 ? 8 : 9, 0xFFB7D7E6);
        panel.addView(label);
        SeekBar bar = new SeekBar(this);
        bar.setMax((max-min)/step);
        if (android.os.Build.VERSION.SDK_INT >= 21) {
            bar.setProgressTintList(ColorStateList.valueOf(0xFF22D3EE));
            bar.setThumbTintList(ColorStateList.valueOf(0xFF22D3EE));
            bar.setProgressBackgroundTintList(ColorStateList.valueOf(0xFF405267));
        }
        panel.addView(bar, new LinearLayout.LayoutParams(-1, dp(29)));
        bar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            public void onProgressChanged(SeekBar b,int progress,boolean user) {
                int value = min + progress*step;
                if (kind==0) {scaleFactor=value/100f;label.setText(name+": "+scaleFactor);}
                if (kind==1) {minNeighbors=value;label.setText(name+": "+value+(value<4?" (possible false positives)":""));}
                if (kind==2) {minFace=value;label.setText(name+": "+value+" px");}
                if (kind==3) {interval=value;label.setText(name+": "+value+" ms");}
                if (kind==4) {holdMs=value;label.setText(name+": "+value+" ms");}
                if (kind==5) {smoothing=value/100f;label.setText(name+": "+value+"%");}
                if (kind==6) {yuNetConfidence=value/100f;label.setText(name+": "+value+"%");}
                if (kind==7) {yuNetNms=value/100f;label.setText(name+": "+value+"%");}
                if (kind==8) {yuNetTopK=value;label.setText(name+": "+value);}
                if (user && kind >= 6) {
                    synchronized (yuNetLock) {
                        if (yuNet != null) {
                            yuNet.setScoreThreshold(yuNetConfidence);
                            yuNet.setNMSThreshold(yuNetNms);
                            yuNet.setTopK(yuNetTopK);
                        }
                    }
                }
                if (user) {
                    saveSettings();
                    if (imageMode && selectedBitmap != null) detectSelectedImage();
                }
            }
            public void onStartTrackingTouch(SeekBar b) {}
            public void onStopTrackingTouch(SeekBar b) {}
        });
        // Initialize after attaching listener, so the label is never blank.
        bar.setProgress(Math.max(0, Math.min(bar.getMax(), (initial-min)/step)));
        // Some Android versions skip the callback when progress remains zero.
        int current = min + bar.getProgress() * step;
        if (kind == 0) label.setText(name + ": " + (current / 100f));
        else if (kind == 1) label.setText(name + ": " + current + (current < 4 ? " (possible false positives)" : ""));
        else if (kind == 2) label.setText(name + ": " + current + " px");
        else if (kind == 3 || kind == 4) label.setText(name + ": " + current + " ms");
        else if (kind == 8) label.setText(name + ": " + current);
        else label.setText(name + ": " + current + "%");
        label.setTextSize(14);
        label.setMinHeight(dp(21));
    }
    private void createSettings() {
        settingsPanel = new LinearLayout(this);
        settingsPanel.setOrientation(LinearLayout.VERTICAL);
        settingsPanel.setPadding(dp(12), dp(6), dp(12), dp(6));
        settingsPanel.setBackground(rounded(0xFF142236, 0xFF2A4159, 20));
        TextView heading = new TextView(this);
        heading.setText("Face detection settings (live)");
        setIcon(heading, 10, 0xFFE2EAF4);
        heading.setTextSize(16);
        heading.setTextColor(Color.WHITE);
        heading.setPadding(dp(5), dp(2), dp(5), dp(3));
        settingsPanel.addView(heading);
        makeSlider(settingsPanel,"Confidence threshold",Math.round(yuNetConfidence*100),10,5,95,6);
        makeSlider(settingsPanel,"NMS threshold",Math.round(yuNetNms*100),5,5,90,7);
        makeSlider(settingsPanel,"Max detections",yuNetTopK,100,100,2000,8);
        makeSlider(settingsPanel,"Interval",interval,50,50,1000,3);
        makeSlider(settingsPanel,"Hold time",holdMs,0,50,2000,4);
        makeSlider(settingsPanel,"Smoothing",Math.round(smoothing*100),0,5,95,5);
        android.widget.ScrollView scroll = new android.widget.ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setVerticalScrollBarEnabled(true);
        scroll.setSmoothScrollingEnabled(true);
        scroll.addView(settingsPanel);
        scroll.setBackgroundColor(0xFF091420);
        scroll.setPadding(dp(10), dp(3), dp(10), dp(4));
        scroll.setClipToPadding(false);
        // Dedicated lower region: never overlays the image or action buttons.
        // Scroll only the settings. The preview and action buttons remain fixed.
        LinearLayout.LayoutParams settingsLp = new LinearLayout.LayoutParams(-1, 0, 1.55f);
        root.addView(scroll, settingsLp);
    }
    private int dp(int value) {
        return (int)(value * getResources().getDisplayMetrics().density + 0.5f);
    }
    private void updatePreviewLayout() {
        if (surface == null || overlay == null || previewArea == null) return;
        int sw = previewArea.getWidth();
        int sh = previewArea.getHeight();
        if (sw <= 0 || sh <= 0) return;
        int iw = (frameRotation == 90 || frameRotation == 270) ? height : width;
        int ih = (frameRotation == 90 || frameRotation == 270) ? width : height;
        if (iw <= 0 || ih <= 0) return;
        float factor = Math.min((float) sw / iw, (float) sh / ih);
        int vw = Math.max(1, Math.round(iw * factor));
        int vh = Math.max(1, Math.round(ih * factor));
        FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(vw, vh, Gravity.CENTER);
        surface.setLayoutParams(lp);
        overlay.setLayoutParams(new FrameLayout.LayoutParams(vw, vh, Gravity.CENTER));
        // ImageFaceView fits the bitmap inside its bounds using uniform scaling.
        imageView.setLayoutParams(new FrameLayout.LayoutParams(-1, -1));
    }
    private RectF[] stabilize(RectF[] next) {
        long now=System.currentTimeMillis();
        if(next.length==0) {
            if(now-lastDetected<=holdMs) return displayed;
            displayed=new RectF[0];return displayed;
        }
        lastDetected=now;
        if(next.length==1 && displayed.length==1) {
            RectF a=displayed[0], b=next[0];
            float cx=(a.centerX()-b.centerX()),cy=(a.centerY()-b.centerY());
            if(cx*cx+cy*cy<0.09f) {
                float k=1f-smoothing;
                next[0]=new RectF(a.left*(1-k)+b.left*k,a.top*(1-k)+b.top*k,
                    a.right*(1-k)+b.right*k,a.bottom*(1-k)+b.bottom*k);
            }
        }
        displayed=next;return displayed;
    }

    public void onCreate(Bundle b) {
        super.onCreate(b);
        prefs=getSharedPreferences("face_settings",MODE_PRIVATE);
        scaleFactor=prefs.getFloat("scale",1.10f);minNeighbors=prefs.getInt("neighbors",5);
        minFace=prefs.getInt("face",70);interval=prefs.getInt("interval",150);
        holdMs=prefs.getInt("hold",400);smoothing=prefs.getFloat("smooth",0.5f);
        yuNetConfidence=prefs.getFloat("yunet_conf",0.70f);
        yuNetNms=prefs.getFloat("yunet_nms",0.30f);
        yuNetTopK=prefs.getInt("yunet_topk",500);
        meshEnabled=prefs.getBoolean("mesh_enabled",false);
        root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(0xFF091420);
        status = new TextView(this);
        status.setTextColor(Color.WHITE);
        status.setTextSize(14);
        status.setBackgroundColor(0xFF202020);
        status.setPadding(dp(12), dp(8), dp(12), dp(8));
        status.setText("Starting OpenCV...");
        // Keep diagnostics available for error handling without occupying the top of the screen.
        status.setVisibility(View.GONE);

        TextView appTitle = new TextView(this);
        appTitle.setText("Face Detector");
        appTitle.setTextColor(Color.WHITE);
        appTitle.setTextSize(23);
        appTitle.setTypeface(null, android.graphics.Typeface.BOLD);
        appTitle.setGravity(Gravity.CENTER_VERTICAL);
        appTitle.setPadding(dp(18), dp(9), dp(16), dp(7));
        root.addView(appTitle, new LinearLayout.LayoutParams(-1, -2));

        LinearLayout actions = new LinearLayout(this);
        actions.setOrientation(LinearLayout.HORIZONTAL);
        actions.setPadding(dp(7), 0, dp(7), 0);
        loadButton = new Button(this);
        loadButton.setText("Load image");
        styleAction(loadButton, false);
        setIcon(loadButton, 0, 0xFFE8F2FF);
        actions.addView(loadButton, buttonParams());
        cameraButton = new Button(this);
        cameraButton.setText("Camera");
        styleAction(cameraButton, false);
        setIcon(cameraButton, 1, 0xFFE8F2FF);
        actions.addView(cameraButton, buttonParams());
        root.addView(actions, new LinearLayout.LayoutParams(-1, -2));

        LinearLayout modes = new LinearLayout(this);
        modes.setOrientation(LinearLayout.HORIZONTAL);
        modes.setPadding(dp(7), 0, dp(7), dp(2));
        Button switchCameraButton = new Button(this);
        switchCameraButton.setText("Switch camera");
        styleAction(switchCameraButton, false);
        setIcon(switchCameraButton, 2, 0xFFE8F2FF);
        modes.addView(switchCameraButton, buttonParams());
        switchCameraButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) { switchCamera(); }
        });
        meshButton = new Button(this);
        meshButton.setText(meshEnabled ? "Face mesh  ON" : "Face mesh  OFF");
        styleAction(meshButton, meshEnabled);
        setIcon(meshButton, 3, meshEnabled ? 0xFF06131E : 0xFFE8F2FF);
        modes.addView(meshButton, buttonParams());
        root.addView(modes, new LinearLayout.LayoutParams(-1, -2));
        meshButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                meshEnabled = !meshEnabled;
                prefs.edit().putBoolean("mesh_enabled", meshEnabled).apply();
                meshButton.setText(meshEnabled ? "Face mesh  ON" : "Face mesh  OFF");
                styleAction(meshButton, meshEnabled);
        setIcon(meshButton, 3, meshEnabled ? 0xFF06131E : 0xFFE8F2FF);
                if (overlay != null) {
                    overlay.meshEnabled = meshEnabled;
                    overlay.invalidate();
                }
                if (imageMode && selectedBitmap != null) detectSelectedImage();
            }
        });
        previewArea = new FrameLayout(this);
        previewArea.setBackground(rounded(0xFF020B12, 0xFF2A4159, 16));
        previewArea.setClipToOutline(true);
        LinearLayout.LayoutParams previewLp = new LinearLayout.LayoutParams(-1, 0, 0.85f);
        previewLp.setMargins(dp(10), dp(4), dp(10), dp(3));
        root.addView(previewArea, previewLp);
        surface = new SurfaceView(this);
        previewArea.addView(surface);
        overlay = new Overlay(this);
        overlay.meshEnabled = meshEnabled;
        previewArea.addView(overlay);
        imageView = new ImageFaceView(this);
        imageView.setVisibility(View.GONE);
        previewArea.addView(imageView, new FrameLayout.LayoutParams(-1, -1));
        previewArea.addOnLayoutChangeListener(new View.OnLayoutChangeListener() {
            public void onLayoutChange(View v, int left, int top, int right, int bottom,
                int oldLeft, int oldTop, int oldRight, int oldBottom) {
                if (right-left != oldRight-oldLeft || bottom-top != oldBottom-oldTop) updatePreviewLayout();
            }
        });
        loadButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) { pickImage(); }
        });
        cameraButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) { showCamera(); }
        });
        createSettings();
        setContentView(root);
        surface.getHolder().addCallback(this);

        try {
            System.loadLibrary("opencv_java4");
            opencvReady = true;
        } catch (Throwable e) {
            status.setText("OpenCV load failed. Check native libraries");
        }
        if (opencvReady) {
            try { ensureBundledYuNetModel(); }
            catch (Throwable e) { status.setText("Classifier error: " + e.getMessage()); }
        }
        if (android.os.Build.VERSION.SDK_INT >= 23 && checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[] {Manifest.permission.CAMERA}, 101);
        }
    }

    // Copy the bundled ONNX model into internal storage because FaceDetectorYN
    // requires a filesystem path rather than an Android asset stream.
    private void ensureBundledYuNetModel() throws IOException {
        File target = new File(getFilesDir(), "yunet.onnx");
        InputStream input = getAssets().open("face_detection_yunet_2026may.onnx");
        File temp = new File(getFilesDir(), "yunet_bundled.tmp");
        try {
            FileOutputStream output = new FileOutputStream(temp);
            try {
                byte[] buffer = new byte[16384];
                int count;
                while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
            } finally { output.close(); }
        } finally { input.close(); }
        if (temp.length() < 100000) { temp.delete(); throw new IOException("Bundled model is incomplete"); }
        if (target.exists() && target.length() == temp.length()) { temp.delete(); return; }
        synchronized (yuNetLock) {
            if (target.exists() && !target.delete()) throw new IOException("Cannot replace model");
            if (!temp.renameTo(target)) throw new IOException("Cannot install bundled model");
            yuNet = null;
        }
    }

    private void loadClassifier() throws Exception {
        File f = new File(getFilesDir(), "haarcascade_frontalface_default.xml");
        InputStream in = getAssets().open("haarcascade_frontalface_default.xml");
        FileOutputStream out = new FileOutputStream(f);
        try {
            byte[] buf = new byte[8192]; int n;
            while ((n = in.read(buf)) != -1) out.write(buf, 0, n);
        } finally { try { in.close(); } catch (Exception ignored) {} try { out.close(); } catch (Exception ignored) {} }
        classifier = new CascadeClassifier(f.getAbsolutePath());
        if (classifier.empty()) throw new Exception("Cannot load Haar XML");
    }

    public void onRequestPermissionsResult(int req, String[] p, int[] grants) {
        super.onRequestPermissionsResult(req, p, grants);
        if (req == 101 && grants.length > 0 && grants[0] == PackageManager.PERMISSION_GRANTED) startCamera();
        else status.setText("Camera permission required");
    }

    private int chooseCamera(boolean front) {
        Camera.CameraInfo info = new Camera.CameraInfo();
        for (int i = 0; i < Camera.getNumberOfCameras(); i++) {
            Camera.getCameraInfo(i, info);
            if (info.facing == (front ? Camera.CameraInfo.CAMERA_FACING_FRONT : Camera.CameraInfo.CAMERA_FACING_BACK)) return i;
        }
        return -1;
    }

    private void switchCamera() {
        if (imageMode) return;
        int front = chooseCamera(true), back = chooseCamera(false);
        int next = cameraId == back ? front : back;
        if (next < 0) next = back;
        stopCamera(); cameraId = next; startCamera();
    }

    private void startCamera() {
        if (imageMode || !opencvReady) return;
        if (camera != null || !surface.getHolder().getSurface().isValid()) return;
        if (android.os.Build.VERSION.SDK_INT >= 23 && checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return;
        try {
            if (cameraId < 0) cameraId = chooseCamera(false);
            if (cameraId < 0) throw new Exception("No camera available");
            camera = Camera.open(cameraId);
            Camera.Parameters params = camera.getParameters();
            List<Camera.Size> sizes = params.getSupportedPreviewSizes();
            Camera.Size chosen = sizes.get(0);
            int best = Integer.MAX_VALUE;
            for (Camera.Size s : sizes) {
                int score = Math.abs(s.width - 640) + Math.abs(s.height - 480);
                if (score < best) { best = score; chosen = s; }
            }
            width = chosen.width; height = chosen.height;
            params.setPreviewSize(width, height);
            params.setPreviewFormat(android.graphics.ImageFormat.NV21);
            camera.setParameters(params);
            Camera.CameraInfo info = new Camera.CameraInfo();
            Camera.getCameraInfo(cameraId, info);
            int degrees = 0;
            int displayRotation = getWindowManager().getDefaultDisplay().getRotation();
            if (displayRotation == 1) degrees = 90;
            if (displayRotation == 2) degrees = 180;
            if (displayRotation == 3) degrees = 270;
            if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                rotation = (info.orientation + degrees) % 360;
                camera.setDisplayOrientation((360 - rotation) % 360);
                rotation = (360 - rotation) % 360;
            } else {
                rotation = (info.orientation - degrees + 360) % 360;
                camera.setDisplayOrientation(rotation);
            }
            frontFacing = info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT;
            overlay.front = frontFacing;
            frameRotation=rotation;
            updatePreviewLayout();
            camera.setPreviewDisplay(surface.getHolder());
            camera.setPreviewCallback(this);
            camera.startPreview();
            status.setText("Faces: 0   |   Tap to switch camera");
        } catch (Throwable e) { stopCamera(); status.setText("Camera error: " + e.getMessage()); }
    }

    public void onPreviewFrame(byte[] data, Camera c) {
        if (imageMode || busy || System.currentTimeMillis() - lastScan < interval) return;
        busy = true; lastScan = System.currentTimeMillis();
        Mat yuv = null, gray = null, upright = null, small = null; MatOfRect faces = null;
        try {
            yuv = new Mat(height + height / 2, width, CvType.CV_8UC1);
            yuv.put(0, 0, data);
            gray = new Mat();
            Imgproc.cvtColor(yuv, gray, Imgproc.COLOR_YUV2GRAY_NV21);
            upright = new Mat();
            if (rotation == 90) Core.rotate(gray, upright, Core.ROTATE_90_CLOCKWISE);
            else if (rotation == 180) Core.rotate(gray, upright, Core.ROTATE_180);
            else if (rotation == 270) Core.rotate(gray, upright, Core.ROTATE_90_COUNTERCLOCKWISE);
            else gray.copyTo(upright);
            small = new Mat();
            double scale = Math.min(1.0, 480.0 / upright.cols());
            Imgproc.resize(upright, small, new Size(), scale, scale, Imgproc.INTER_LINEAR);
            if (!useYuNet) Imgproc.equalizeHist(small, small);
            double[][] found;
            if (useYuNet) {
                Mat bgr = new Mat();
                try {
                    Imgproc.cvtColor(small, bgr, Imgproc.COLOR_GRAY2BGR);
                    found = detectYuNet(bgr);
                } finally { bgr.release(); }
            } else {
                faces = new MatOfRect();
                synchronized (classifier) { classifier.detectMultiScale(small, faces, scaleFactor, minNeighbors, 0, new Size(Math.max(1,minFace*scale), Math.max(1,minFace*scale)), new Size()); }
                found = readFaceRects(faces);
            }
            final StringBuilder cameraDebug = new StringBuilder();
            cameraDebug.append("Detect ").append(small.cols()).append("x").append(small.rows())
                .append(" | upright ").append(upright.cols()).append("x").append(upright.rows());
            for (int i = 0; i < Math.min(found.length, 3); i++) {
                double[] r = found[i];
                cameraDebug.append(String.format(java.util.Locale.US,
                    "\n#%d raw [%.0f,%.0f,%.0f,%.0f]", i + 1, r[0], r[1], r[2], r[3]));
            }
            if (useYuNet && latestYuNetRows.length > 0) {
                double[] r = latestYuNetRows[0];
                cameraDebug.append(String.format(java.util.Locale.US,
                    "\nEyes (%.0f,%.0f) (%.0f,%.0f) nose (%.0f,%.0f) mouth (%.0f,%.0f) (%.0f,%.0f)",
                    r[4],r[5],r[6],r[7],r[8],r[9],r[10],r[11],r[12],r[13]));
            }
            cameraDebug.append(" | front=").append(frontFacing).append(" rotation=").append(rotation)
                .append(" | Y flip=").append(frontFacing);
            final RectF[] rects = new RectF[found.length];
            for (int i = 0; i < found.length; i++) {
                double[] r = found[i];
                // Front preview is vertically inverted relative to the rotated NV21
                // detector matrix on this device. Keep X unchanged (v10.0 fix),
                // and invert Y in detector coordinates before normalization.
                float left = (float)(r[0] / small.cols());
                float right = (float)((r[0] + r[2]) / small.cols());
                float top = frontFacing
                        ? (float)((small.rows() - r[1] - r[3]) / small.rows())
                        : (float)(r[1] / small.rows());
                float bottom = frontFacing
                        ? (float)((small.rows() - r[1]) / small.rows())
                        : (float)((r[1] + r[3]) / small.rows());
                rects[i] = new RectF(left, top, right, bottom);
            }
            // The detector frame is vertically inverted relative to the front
            // preview. Flip the model crop before inference, then map its
            // landmarks back into detector coordinates and preview coordinates.
            final float[][] meshPoints = meshEnabled && useYuNet
                    ? estimateMeshes(small, found, frontFacing) : new float[0][];
            runOnUiThread(new Runnable() {
                public void run() {
                    overlay.meshPoints = meshPoints;
                    overlay.rects = stabilize(rects);
                    overlay.meshEnabled = meshEnabled;
                    overlay.invalidate();
                    status.setText("YuNet faces: " + overlay.rects.length + " | Tap to switch camera"
                        + (meshEnabled ? " | Mesh: " + meshDiagnostic : ""));
                }
            });
        } catch (Throwable e) {
            final String msg = e.getMessage();
            runOnUiThread(new Runnable() { public void run() { status.setText("Detection error: " + msg); } });
        } finally {
            if (faces != null) faces.release(); if (small != null) small.release();
            if (upright != null) upright.release(); if (gray != null) gray.release(); if (yuv != null) yuv.release();
            busy = false;
        }
    }

    private void stopCamera() {
        if (camera != null) {
            try { camera.setPreviewCallback(null); camera.stopPreview(); camera.release(); } catch (Exception ignored) {}
            camera = null;
        }
        overlay.rects = new RectF[0]; displayed=new RectF[0]; overlay.invalidate();
    }
    public void surfaceCreated(SurfaceHolder h) { startCamera(); }
    public void surfaceChanged(SurfaceHolder h, int format, int w, int ht) { if (camera == null) startCamera(); }
    public void surfaceDestroyed(SurfaceHolder h) { stopCamera(); }
    protected void onPause() { stopCamera(); super.onPause(); }
    protected void onResume() { super.onResume(); if (!imageMode && surface != null) startCamera(); }
    protected void onDestroy() {
        stopCamera();
        imageGeneration++;
        if (selectedBitmap != null) { selectedBitmap.recycle(); selectedBitmap = null; }
        super.onDestroy();
    }


    private void pickImage() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.setType("image/*");
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        try { startActivityForResult(intent, PICK_IMAGE); }
        catch (Exception e) { status.setText("Cannot open image picker: " + e.getMessage()); }
    }

    private void showCamera() {
        imageGeneration++;
        imageMode = false;
        imageView.setVisibility(View.GONE);
        surface.setVisibility(View.VISIBLE);
        overlay.setVisibility(View.VISIBLE);
        if (selectedBitmap != null) { selectedBitmap.recycle(); selectedBitmap = null; }
        status.setText("Starting camera...");
        startCamera();
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request != PICK_IMAGE || result != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        Bitmap bitmap = null;
        try {
            BitmapFactory.Options opts = new BitmapFactory.Options();
            opts.inJustDecodeBounds = true;
            InputStream input = getContentResolver().openInputStream(uri);
            try { BitmapFactory.decodeStream(input, null, opts); } finally { if (input != null) input.close(); }
            int sample = 1;
            while (opts.outWidth / sample > 1600 || opts.outHeight / sample > 1600) sample *= 2;
            opts = new BitmapFactory.Options();
            opts.inSampleSize = sample;
            input = getContentResolver().openInputStream(uri);
            try { bitmap = BitmapFactory.decodeStream(input, null, opts); }
            finally { if (input != null) input.close(); }
            if (bitmap == null) throw new IOException("Unsupported image");
            int orientation = ExifInterface.ORIENTATION_NORMAL;
            if (android.os.Build.VERSION.SDK_INT >= 24) {
                input = getContentResolver().openInputStream(uri);
                try { orientation = new ExifInterface(input).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL); }
                finally { if (input != null) input.close(); }
            }
            Matrix transform = new Matrix();
            if (orientation == ExifInterface.ORIENTATION_ROTATE_90) transform.postRotate(90);
            else if (orientation == ExifInterface.ORIENTATION_ROTATE_180) transform.postRotate(180);
            else if (orientation == ExifInterface.ORIENTATION_ROTATE_270) transform.postRotate(270);
            else if (orientation == ExifInterface.ORIENTATION_FLIP_HORIZONTAL) transform.postScale(-1, 1);
            else if (orientation == ExifInterface.ORIENTATION_TRANSPOSE) { transform.postRotate(90); transform.postScale(-1, 1); }
            else if (orientation == ExifInterface.ORIENTATION_TRANSVERSE) { transform.postRotate(270); transform.postScale(-1, 1); }
            else if (orientation == ExifInterface.ORIENTATION_FLIP_VERTICAL) transform.postScale(1, -1);
            if (!transform.isIdentity()) {
                Bitmap rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), transform, true);
                if (rotated != bitmap) { bitmap.recycle(); bitmap = rotated; }
            }
            imageGeneration++;
            stopCamera();
            imageMode = true;
            surface.setVisibility(View.GONE);
            overlay.setVisibility(View.GONE);
            imageView.setVisibility(View.VISIBLE);
            if (selectedBitmap != null) selectedBitmap.recycle();
            selectedBitmap = bitmap;
            imageView.setImage(bitmap);
            status.setText("Detecting image...");
            detectSelectedImage();
        } catch (Throwable e) {
            if (bitmap != null && bitmap != selectedBitmap) bitmap.recycle();
            status.setText("Image load error: " + e.getMessage());
        }
    }

    // YuNet returns a CV_32F matrix with 15 columns per detection:
    // x, y, width, height, 5 landmark pairs, confidence.
    private volatile double[][] latestYuNetRows = new double[0][];
    private double[][] detectYuNet(Mat bgr) throws Exception {
        synchronized (yuNetLock) {
            File model = new File(getFilesDir(), "yunet.onnx");
            if (!model.exists()) ensureBundledYuNetModel();
            if (yuNet == null) yuNet = FaceDetectorYN.create(model.getAbsolutePath(), "",
                new Size(bgr.cols(), bgr.rows()), yuNetConfidence, yuNetNms, yuNetTopK);
            if (yuNet == null) throw new IOException("Cannot initialize YuNet");
            yuNet.setInputSize(new Size(bgr.cols(), bgr.rows()));
            Mat results = new Mat();
            try {
                int detectionResult = yuNet.detect(bgr, results);
                yuNetDiagnostics = "YuNet model=" + model.length() + " bytes"
                    + " | input=" + bgr.cols() + "x" + bgr.rows()
                    + " ch=" + bgr.channels()
                    + " | return=" + detectionResult
                    + " | output=" + results.rows() + "x" + results.cols()
                    + " type=" + results.type()
                    + " | threshold=" + yuNetConfidence
                    + " | NMS=" + yuNetNms + " | topK=" + yuNetTopK;
                java.util.ArrayList<double[]> found = new java.util.ArrayList<double[]>();
                if (results.rows() > 0 && results.cols() >= 15) {
                    double[] confidence = results.get(0, 14);
                    if (confidence != null && confidence.length > 0)
                        yuNetDiagnostics += String.format(java.util.Locale.US,
                            " | first confidence=%.3f", confidence[0]);
                }
                java.util.ArrayList<double[]> allRows = new java.util.ArrayList<double[]>();
                for (int i = 0; i < results.rows(); i++) {
                    double[] row = new double[15];
                    for (int j = 0; j < 15; j++) {
                        double[] cell = results.get(i, j);
                        if (cell == null || cell.length == 0) throw new IOException("Invalid YuNet result");
                        row[j] = cell[0];
                    }
                    if (row[2] > 0 && row[3] > 0) {
                        found.add(new double[] {row[0],row[1],row[2],row[3]});
                        allRows.add(row);
                    }
                }
                latestYuNetRows = allRows.toArray(new double[allRows.size()][]);
                return found.toArray(new double[found.size()][]);
            } finally { results.release(); }
        }
    }

    // Read raw CV_32SC4 values without constructing org.opencv.core.Rect.
    // The supplied OpenCV JAR's Rect object fields can be zero even when
    // Mat.get() returns valid native rectangle coordinates.
    private static double[][] readFaceRects(MatOfRect faces) {
        java.util.ArrayList<double[]> valid = new java.util.ArrayList<double[]>();
        int rows = faces.rows();
        int cols = faces.cols();
        if (cols <= 0) return new double[0][];
        int count = Math.min(rows * cols, 10000);
        for (int i = 0; i < count; i++) {
            double[] v = faces.get(i / cols, i % cols);
            if (v == null || v.length < 4) continue;
            if (Double.isNaN(v[0]) || Double.isNaN(v[1]) ||
                Double.isNaN(v[2]) || Double.isNaN(v[3])) continue;
            if (v[0] >= 0 && v[1] >= 0 && v[2] > 0 && v[3] > 0) {
                valid.add(new double[] {v[0], v[1], v[2], v[3]});
            }
        }
        return valid.toArray(new double[valid.size()][]);
    }

    private void detectSelectedImage() {
        if (!imageMode || selectedBitmap == null) return;
        if (!opencvReady) { status.setText("OpenCV unavailable"); return; }
        final Bitmap bitmap = selectedBitmap;
        final int generation = ++imageGeneration;
        final double factor = scaleFactor;
        final int neighbors = minNeighbors, minSize = minFace;
        new Thread(new Runnable() {
            public void run() {
                Mat rgba = null, gray = null, small = null;
                MatOfRect faces = null;
                try {
                    // Convert Bitmap to RGBA Mat without org.opencv.android.Utils.
                    // Use Java 7-compatible Android Bitmap APIs and Mat.put().
                    int bw = bitmap.getWidth();
                    int bh = bitmap.getHeight();
                    int[] pixels = new int[bw * bh];
                    bitmap.getPixels(pixels, 0, bw, 0, 0, bw, bh);
                    byte[] rgbaBytes = new byte[bw * bh * 4];
                    for (int p = 0; p < pixels.length; p++) {
                        int color = pixels[p];
                        int offset = p * 4;
                        rgbaBytes[offset] = (byte) ((color >> 16) & 255);
                        rgbaBytes[offset + 1] = (byte) ((color >> 8) & 255);
                        rgbaBytes[offset + 2] = (byte) (color & 255);
                        rgbaBytes[offset + 3] = (byte) ((color >>> 24) & 255);
                    }
                    rgba = new Mat(bh, bw, CvType.CV_8UC4);
                    rgba.put(0, 0, rgbaBytes);
                    gray = new Mat();
                    Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY);
                    small = new Mat();
                    double resize = Math.min(1.0, 800.0 / Math.max(gray.cols(), gray.rows()));
                    Imgproc.resize(gray, small, new Size(), resize, resize, Imgproc.INTER_LINEAR);
                    if (!useYuNet) Imgproc.equalizeHist(small, small);
                    double[][] found;
                    Mat detectorFrame = null;
                    if (useYuNet) {
                        Mat bgr = new Mat();
                        try {
                            Mat color = new Mat();
                            try {
                                Imgproc.cvtColor(rgba, color, Imgproc.COLOR_RGBA2BGR);
                                Imgproc.resize(color, bgr, new Size(small.cols(), small.rows()));
                                found = detectYuNet(bgr);
                                detectorFrame = new Mat();
                                bgr.copyTo(detectorFrame);
                            } finally { color.release(); }
                        } finally { bgr.release(); }
                    } else {
                        faces = new MatOfRect();
                        synchronized (classifier) {
                            classifier.detectMultiScale(small, faces, factor, neighbors, 0,
                                new Size(Math.max(1, minSize * resize), Math.max(1, minSize * resize)), new Size());
                        }
                        found = readFaceRects(faces);
                        detectorFrame = new Mat();
                        small.copyTo(detectorFrame);
                    }
                    // v9.7: Draw directly onto OpenCV detector Mat, BEFORE Android rendering.
                    // Green = raw rectangle, red cross = raw top-left coordinate.
                    // This isolates native detector coordinates from ImageFaceView overlays.
                    if (detectorFrame != null) {
                        Scalar green = detectorFrame.channels() == 1 ? new Scalar(255) : new Scalar(0, 255, 0);
                        Scalar cross = detectorFrame.channels() == 1 ? new Scalar(150) : new Scalar(0, 0, 255);
                        for (int j = 0; j < found.length; j++) {
                            double[] r = found[j];
                            double x = r[0], y = r[1], w = r[2], h = r[3];
                            Imgproc.rectangle(detectorFrame, new Point(x, y), new Point(x + w, y + h), green, 3);
                            if (meshEnabled && useYuNet) {
                                float[][] points = estimateMeshes(detectorFrame,
                                    new double[][] {r}, false);
                                if (points.length > 0) drawModelMesh(detectorFrame, points[0]);
                            }
                            // Crosshair diagnostics removed.

                        }
                    }
                    // Build a bitmap from the exact pixels supplied to the detector.
                    // Avoid MatOfRect/Rect conversions and any original-photo mapping.
                    final Bitmap detectorBitmap;
                    try {
                        int dw = detectorFrame.cols(), dh = detectorFrame.rows();
                        byte[] rgb = new byte[dw * dh * detectorFrame.channels()];
                        detectorFrame.get(0, 0, rgb);
                        int[] argb = new int[dw * dh];
                        if (detectorFrame.channels() == 1) {
                            for (int k = 0; k < argb.length; k++) {
                                int v = rgb[k] & 255;
                                argb[k] = Color.rgb(v, v, v);
                            }
                        } else {
                            for (int k = 0; k < argb.length; k++) {
                                int b = rgb[k * 3] & 255;
                                int g = rgb[k * 3 + 1] & 255;
                                int r = rgb[k * 3 + 2] & 255;
                                argb[k] = Color.rgb(r, g, b);
                            }
                        }
                        detectorBitmap = Bitmap.createBitmap(argb, dw, dh, Bitmap.Config.ARGB_8888);
                    } finally { if (detectorFrame != null) detectorFrame.release(); }
                    final RectF[] boxes = new RectF[found.length];
                    final int rawFaceCount = faces == null ? found.length : (int)faces.total();
                    final int matType = faces == null ? 0 : faces.type();
                    final int matRows = faces == null ? 0 : faces.rows();
                    final int matCols = faces == null ? 0 : faces.cols();
                    final int matChannels = faces == null ? 0 : faces.channels();
                    final double[] rawFirst = faces != null && rawFaceCount > 0 ? faces.get(0, 0) : null;
                    for (int i = 0; i < found.length; i++) {
                        double[] r = found[i];
                        boxes[i] = new RectF((float)(r[0] / small.cols()), (float)(r[1] / small.rows()),
                            (float)((r[0]+r[2]) / small.cols()), (float)((r[1]+r[3]) / small.rows()));
                    }
                    final StringBuilder imageDebug = new StringBuilder();
                    imageDebug.append("Bitmap ").append(bw).append("x").append(bh)
                        .append(" | detect ").append(small.cols()).append("x").append(small.rows())
                        .append(String.format(java.util.Locale.US, " | resize %.4f", resize));
                    if (faces != null) {
                        imageDebug.append(" | Mat ").append(matRows).append("x").append(matCols)
                            .append(" ch=").append(matChannels).append(" type=").append(matType);
                    }
                    for (int i = 0; i < Math.min(found.length, 5); i++) {
                        double[] r = found[i];
                        RectF b = boxes[i];
                        imageDebug.append(String.format(java.util.Locale.US,
                            "\n#%d raw [%.0f,%.0f,%.0f,%.0f] -> img [%.0f,%.0f,%.0f,%.0f]",
                            i + 1, r[0],r[1],r[2],r[3],
                            b.left*bw,b.top*bh,b.right*bw,b.bottom*bh));
                    }
                    if (useYuNet && latestYuNetRows.length > 0) {
                        double[] r = latestYuNetRows[0];
                        imageDebug.append(String.format(java.util.Locale.US,
                            "\nLandmarks eyes (%.0f,%.0f) (%.0f,%.0f) nose (%.0f,%.0f) mouth (%.0f,%.0f) (%.0f,%.0f)",
                            r[4],r[5],r[6],r[7],r[8],r[9],r[10],r[11],r[12],r[13]));
                    }
                    runOnUiThread(new Runnable() {
                        public void run() {
                            if (imageMode && generation == imageGeneration) {
                                // Draw using the SAME view that displays the photograph.
                                // Diagnostic yellow frame is deliberately independent of detections.
                                imageView.setImage(detectorBitmap);
                                imageView.setFaces(new RectF[0]); // Already drawn on native Mat.
                                imageView.setDiagnostic(false);
                                status.setText("YuNet image faces: " + boxes.length);
                            }
                        }
                    });
                } catch (final Throwable e) {
                    runOnUiThread(new Runnable() {
                        public void run() { if (generation == imageGeneration) status.setText("Image detection error: " + e.getMessage()); }
                    });
                } finally {
                    if (faces != null) faces.release();
                    if (small != null) small.release();
                    if (gray != null) gray.release();
                    if (rgba != null) rgba.release();
                }
            }
        }, "ImageFaceDetection").start();
    }

    private static class ImageFaceView extends View {
        private Bitmap bitmap;
        private RectF[] faces = new RectF[0];
        private boolean diagnostic = false;
        private Paint imagePaint = new Paint(3);
        private Paint boxPaint = new Paint(3);
        ImageFaceView(Context context) {
            super(context);
            setLayerType(View.LAYER_TYPE_SOFTWARE, null);
            boxPaint.setColor(Color.GREEN);
            boxPaint.setStyle(Paint.Style.STROKE);
            boxPaint.setStrokeWidth(6f);
            boxPaint.setAntiAlias(true);
        }
        void setImage(Bitmap value) {
            bitmap = value;
            faces = new RectF[0];
            invalidate();
        }
        void setDiagnostic(boolean enabled) { diagnostic = enabled; invalidate(); }
        void setFaces(RectF[] value) {
            faces = value == null ? new RectF[0] : value;
            invalidate();
        }
        @Override protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            if (bitmap == null || bitmap.isRecycled()) return;
            int vw = getWidth(), vh = getHeight();
            if (vw <= 0 || vh <= 0) return;
            float bw = bitmap.getWidth(), bh = bitmap.getHeight();
            float factor = Math.min(vw / bw, vh / bh);
            float left = (vw - bw * factor) / 2f;
            float top = (vh - bh * factor) / 2f;
            canvas.save();
            canvas.translate(left, top);
            canvas.scale(factor, factor);
            canvas.drawBitmap(bitmap, 0f, 0f, imagePaint);
            if (diagnostic) {
                // Yellow inset border: proves that the custom image view redraws.
                Paint test = new Paint(Paint.ANTI_ALIAS_FLAG);
                test.setColor(Color.YELLOW);
                test.setStyle(Paint.Style.STROKE);
                test.setStrokeWidth(Math.max(5f / factor, 3f));
                canvas.drawRect(bw * 0.06f, bh * 0.06f, bw * 0.94f, bh * 0.94f, test);
            }

            boxPaint.setStrokeWidth(Math.max(8f / factor, 4f));
            Paint idPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
            idPaint.setColor(Color.YELLOW);
            idPaint.setTextSize(Math.max(32f / factor, 22f));
            idPaint.setStyle(Paint.Style.FILL);
            for (int i = 0; i < faces.length; i++) {
                RectF r = faces[i];
                float x1 = Math.max(0f, Math.min(bw, r.left * bw));
                float y1 = Math.max(0f, Math.min(bh, r.top * bh));
                float x2 = Math.max(0f, Math.min(bw, r.right * bw));
                float y2 = Math.max(0f, Math.min(bh, r.bottom * bh));
                if (x2 > x1 && y2 > y1) {
                    canvas.drawRect(x1, y1, x2, y2, boxPaint);
                    canvas.drawText("#" + (i + 1), x1 + 3f, Math.max(35f / factor, y1 - 5f), idPaint);
                }
            }
            canvas.restore();
        }
    }

    // MediaPipe-style 468 landmark inference. Output coordinates are relative to
    // a 192x192 crop; Z is model-relative depth, not metric distance.
    private void ensureFaceMeshModel() throws IOException {
        synchronized (meshLock) {
            if (faceMeshNet != null) return;
            File target = new File(getFilesDir(), "face_mesh_468.onnx");
            if (!target.exists() || target.length() < 1000000) {
                InputStream input = getAssets().open("face_mesh_Nx3x192x192.onnx");
                FileOutputStream output = new FileOutputStream(target);
                try {
                    byte[] buffer = new byte[16384]; int count;
                    while ((count = input.read(buffer)) != -1) output.write(buffer,0,count);
                } finally { try { output.close(); } finally { input.close(); } }
            }
            faceMeshNet = Dnn.readNetFromONNX(target.getAbsolutePath());
            if (faceMeshNet.empty()) throw new IOException("3D face mesh model failed to load");
        }
    }

    private float[][] estimateMeshes(Mat source, double[][] faces, boolean flipY) {
        if (!meshEnabled || faces.length == 0) { meshDiagnostic = meshEnabled ? "NO FACE" : "OFF"; return new float[0][]; }
        meshDiagnostic = "RUNNING";
        float[][] meshes = new float[Math.min(faces.length, 3)][];
        try {
            ensureFaceMeshModel();
            for (int f = 0; f < meshes.length; f++) {
                double[] r = faces[f];
                // Expand YuNet rectangle to include forehead, cheeks and jaw.
                double side = Math.max(r[2],r[3]) * 1.55;
                double cx = r[0]+r[2]*0.5, cy = r[1]+r[3]*0.5;
                double x0 = cx-side*0.5, y0 = cy-side*0.5;
                Mat patch = new Mat(192,192,CvType.CV_8UC3,new Scalar(0,0,0));
                Mat resized = new Mat(); Mat rgb = new Mat(); Mat blob = null;
                try {
                    // Clamp crop and preserve its original coordinate transform.
                    int sx=Math.max(0,(int)Math.floor(x0)), sy=Math.max(0,(int)Math.floor(y0));
                    int ex=Math.min(source.cols(),(int)Math.ceil(x0+side));
                    int ey=Math.min(source.rows(),(int)Math.ceil(y0+side));
                    if (ex<=sx || ey<=sy) continue;
                    org.opencv.core.Rect roi=new org.opencv.core.Rect(sx,sy,ex-sx,ey-sy);
                    Mat cropped=source.submat(roi);
                    try { Imgproc.resize(cropped,resized,new Size(
                        Math.max(1,Math.round((ex-sx)*192.0/side)),
                        Math.max(1,Math.round((ey-sy)*192.0/side)))); }
                    finally { cropped.release(); }
                    int dx=Math.max(0,(int)Math.round((sx-x0)*192.0/side));
                    int dy=Math.max(0,(int)Math.round((sy-y0)*192.0/side));
                    int rw=Math.min(resized.cols(),192-dx), rh=Math.min(resized.rows(),192-dy);
                    if (rw<=0 || rh<=0) continue;
                    if (resized.channels()==1) Imgproc.cvtColor(resized,resized,Imgproc.COLOR_GRAY2BGR);
                    Mat dest=patch.submat(new org.opencv.core.Rect(dx,dy,rw,rh));
                    Mat srcPart=resized.submat(new org.opencv.core.Rect(0,0,rw,rh));
                    try {srcPart.copyTo(dest);} finally {dest.release();srcPart.release();}
                    // The front-camera detector image is upside down relative to
                    // the visible preview. The mesh model must see an upright face.
                    if (flipY) org.opencv.core.Core.flip(patch, patch, 0);
                    Imgproc.cvtColor(patch,rgb,Imgproc.COLOR_BGR2RGB);
                    blob=Dnn.blobFromImage(rgb,1.0/255.0,new Size(192,192),new Scalar(0,0,0),false,false);
                    java.util.List<Mat> outs=new java.util.ArrayList<Mat>();
                    synchronized(meshLock) {
                        faceMeshNet.setInput(blob);
                        faceMeshNet.forward(outs,faceMeshNet.getUnconnectedOutLayersNames());
                    }
                    try {
                        if (outs.size()<2) { meshDiagnostic="OUTPUTS="+outs.size(); continue; }
                        Mat landmarks=outs.get(0), confidence=outs.get(1);
                        double[] score=confidence.get(0,0);
                        // The model emits a logit for face presence.
                        if (score==null || score.length==0) { meshDiagnostic="NO SCORE"; continue; }
                        // Face presence is a raw logit, not a 0..1 probability.
                        double probability = 1.0 / (1.0 + Math.exp(-score[0]));
                        if (probability < 0.10) {
                            meshDiagnostic = "FACE PROBABILITY " + String.format(java.util.Locale.US, "%.0f%%", probability * 100.0);
                            continue;
                        }
                        // The ONNX tensor is [1,468,3]. Java Mat.reshape(cn,rows)
                        // cannot reshape a 3-dimensional Mat using this overload.
                        // Read the tensor directly using multi-dimensional indices.
                        if (landmarks.dims() != 3 || landmarks.size(0) != 1
                                || landmarks.size(1) != 468 || landmarks.size(2) != 3) {
                            meshDiagnostic = "LANDMARK SHAPE " + landmarks.dims() + "D / " + landmarks.total();
                            continue;
                        }
                        float[] result = new float[468 * 3];
                        boolean valid = true;
                        for (int k = 0; k < 468; k++) {
                            double[] vx = landmarks.get(new int[] {0,k,0});
                            double[] vy = landmarks.get(new int[] {0,k,1});
                            double[] vz = landmarks.get(new int[] {0,k,2});
                            if (vx == null || vy == null || vz == null
                                    || vx.length == 0 || vy.length == 0 || vz.length == 0) {
                                valid = false; break;
                            }
                            double px = x0 + vx[0] * side / 192.0;
                            // Undo crop flip to recover detector coordinates;
                            // the final preview mapping then flips Y once.
                            double py = y0 + (flipY ? (192.0 - vy[0]) : vy[0]) * side / 192.0;
                            if (Double.isNaN(px) || Double.isNaN(py)
                                    || Double.isInfinite(px) || Double.isInfinite(py)) {
                                valid = false; break;
                            }
                            result[3*k] = (float)(px / source.cols());
                            result[3*k+1] = (float)(flipY ? 1.0 - py / source.rows() : py / source.rows());
                            result[3*k+2] = (float)(vz[0] * side / 192.0 / source.cols());
                        }
                        if (!valid) result = null;
                        meshes[f]=result;
                        meshDiagnostic=result==null ? "INVALID LANDMARKS" : "468 LANDMARKS / "+(MESH_EDGES.length/2)+" EDGES / "+String.format(java.util.Locale.US,"%.0f%%",probability*100.0);
                    } finally {for(Mat m:outs)m.release();}
                } finally {if(blob!=null)blob.release();rgb.release();resized.release();patch.release();}
            }
        } catch(Throwable e) {
            meshDiagnostic="ERROR: "+e.getClass().getSimpleName()+": "+e.getMessage();
        }
        return meshes;
    }

    private static void drawModelMesh(Mat mat, float[] points) {
        if (points==null) return;
        Scalar color=new Scalar(255,255,0);
        for(int i=0;i<MESH_EDGES.length;i+=2) {
            int a=MESH_EDGES[i]*3,b=MESH_EDGES[i+1]*3;
            Imgproc.line(mat,new Point(points[a]*mat.cols(),points[a+1]*mat.rows()),
                new Point(points[b]*mat.cols(),points[b+1]*mat.rows()),color,1);
        }
    }

    private static class Overlay extends View {
        RectF[] rects = new RectF[0]; boolean front = false;
        float[][] meshPoints = new float[0][];
        boolean meshEnabled = false;
        Paint meshPaint = new Paint(3);
        Paint paint = new Paint();
        Overlay(Context c) { super(c); paint.setColor(Color.GREEN); paint.setStyle(Paint.Style.STROKE); paint.setStrokeWidth(5);
            meshPaint.setColor(Color.CYAN); meshPaint.setStrokeWidth(1.5f); meshPaint.setStyle(Paint.Style.STROKE); }
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            for (RectF r : rects) {
                // The Camera SurfaceView already matches the upright detector orientation.
                // Do not mirror the overlay again for the front camera.
                canvas.drawRect(r.left * getWidth(), r.top * getHeight(),
                        r.right * getWidth(), r.bottom * getHeight(), paint);
            }
            if (meshEnabled) {
                for (float[] points : meshPoints) {
                    if (points == null) continue;
                    for(int i=0;i<MESH_EDGES.length;i+=2) {
                        int a=MESH_EDGES[i]*3,b=MESH_EDGES[i+1]*3;
                        canvas.drawLine(points[a]*getWidth(),points[a+1]*getHeight(),
                            points[b]*getWidth(),points[b+1]*getHeight(),meshPaint);
                    }
                }
            }
        }
    }
}
