# Security - Obfuscate everything
-obfuscationdictionary dictionary.txt
-classobfuscationdictionary dictionary.txt
-packageobfuscationdictionary dictionary.txt

# Remove debugging info
-assumenosideeffects class android.util.Log {
    public static boolean isLoggable(java.lang.String, int);
    public static int v(...);
    public static int i(...);
    public static int w(...);
    public static int d(...);
    public static int e(...);
}

# ========== ANTI-REVERSE ENGINEERING ==========

# Obfuscate security-critical classes aggressively
-keep,allowobfuscation class com.orignal.buddylynk.security.** { *; }
-keepclassmembers class com.orignal.buddylynk.security.** {
    private *;
}

# Hide API configuration (URLs, keys)
-assumenosideeffects class com.orignal.buddylynk.data.api.ApiConfig {
    public static final java.lang.String *;
}

# Remove debugging metadata
-keepattributes !LocalVariableTable,!LocalVariableTypeTable

# Encrypt string constants
-adaptclassstrings
-adaptresourcefilenames
-adaptresourcefilecontents

# ========== OBFUSCATION SETTINGS ==========

# Keep essential classes
-keep class com.orignal.buddylynk.data.model.** { *; }
-keep class com.orignal.buddylynk.data.api.** { *; }

# Protect against reflection attacks
-keepattributes *Annotation*
-keepattributes Signature
-keepattributes InnerClasses
-keepattributes EnclosingMethod

# Hide source file names and line numbers
-renamesourcefileattribute ""
-keepattributes SourceFile,LineNumberTable

# Aggressive obfuscation
-overloadaggressively
-repackageclasses ''
-allowaccessmodification
-flattenpackagehierarchy ''
-optimizationpasses 5
-mergeinterfacesaggressively

# Remove unused code
-dontwarn **
-ignorewarnings

# ========== LIBRARY RULES ==========

# OkHttp
-dontwarn okhttp3.**
-dontwarn okio.**
-keep class okhttp3.** { *; }
-keep interface okhttp3.** { *; }

# Retrofit
-dontwarn retrofit2.**
-keep class retrofit2.** { *; }
-keepattributes Exceptions

# Gson
-keep class com.google.gson.** { *; }
-keepattributes *Annotation*
-dontwarn sun.misc.**
-keep class * extends com.google.gson.TypeAdapter
-keep class * implements com.google.gson.TypeAdapterFactory
-keep class * implements com.google.gson.JsonSerializer
-keep class * implements com.google.gson.JsonDeserializer

# Google Play Services
-keep class com.google.android.gms.** { *; }
-dontwarn com.google.android.gms.**

# Firebase
-keep class com.google.firebase.** { *; }
-dontwarn com.google.firebase.**

# AWS SDK
-keep class aws.sdk.kotlin.** { *; }
-dontwarn aws.sdk.kotlin.**

# Credentials API
-keep class androidx.credentials.** { *; }
-keep class com.google.android.libraries.identity.googleid.** { *; }

# WebRTC
-keep class org.webrtc.** { *; }
-dontwarn org.webrtc.**

# Socket.io
-keep class io.socket.** { *; }
-dontwarn io.socket.**

# Kotlin Coroutines
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}
-keepclassmembers class kotlinx.coroutines.** {
    volatile <fields>;
}

# Room Database
-keep class * extends androidx.room.RoomDatabase
-keep @androidx.room.Entity class *
-dontwarn androidx.room.paging.**

# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# If your project uses WebView with JS, uncomment the following
# and specify the fully qualified class name to the JavaScript interface
# class:
#-keepclassmembers class fqcn.of.javascript.interface.for.webview {
#   public *;
#}

# Uncomment this to preserve the line number information for
# debugging stack traces.
#-keepattributes SourceFile,LineNumberTable

# If you keep the line number information, uncomment this to
# hide the original source file name.
#-renamesourcefileattribute SourceFile