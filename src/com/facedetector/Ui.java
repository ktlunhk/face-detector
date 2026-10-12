package com.facedetector;

import android.app.Activity;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.view.Gravity;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

/** Reusable action button styles and hand-drawn vector icons. */
public final class Ui {
    private Ui() {}
    public static final class Style {
        private final Activity activity;
        public Style(Activity activity) { this.activity = activity; }
        private int dp(int value) { return (int)(value * activity.getResources().getDisplayMetrics().density + 0.5f); }
    public GradientDrawable rounded(int color, int stroke, int radius) {
        GradientDrawable bg = new GradientDrawable();
        bg.setColor(color);
        bg.setCornerRadius(dp(radius));
        if (stroke != 0) bg.setStroke(dp(1), stroke);
        return bg;
    }
    // Lightweight vector icons: no emoji fonts, AndroidX, or external resources.
    public Drawable icon(final int type, final int color) {
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
    public void setIcon(TextView view, int type, int color) {
        view.setCompoundDrawables(icon(type, color), null, null, null);
        view.setCompoundDrawablePadding(dp(8));
    }
    public void styleAction(Button button, boolean selected) {
        button.setAllCaps(false);
        button.setTextSize(12);
        button.setTextColor(selected ? 0xFF06131E : 0xFFE8F2FF);
        button.setBackground(rounded(selected ? 0xFF22D3EE : 0xFF1C2C41,
            selected ? 0xFF22D3EE : 0xFF314A66, 16));
        button.setPadding(dp(4), dp(5), dp(4), dp(5));
        button.setMinHeight(dp(40));
        button.setGravity(Gravity.CENTER);
    }
    public LinearLayout.LayoutParams buttonParams() {
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, dp(42), 1f);
        lp.setMargins(dp(4), dp(3), dp(4), dp(3));
        return lp;
    }

    }
}
