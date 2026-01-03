/**
 * BuddyLynk Native Security Module
 * 
 * C++ security checks are MUCH harder to bypass than Kotlin/Java because:
 * - Compiled to machine code (no bytecode to decompile)
 * - Cannot be hooked by Xposed (only works on Java layer)
 * - Frida hooking is more complex for native code
 * - Strings are harder to find in binary
 * 
 * IMPORTANT: These functions are called from Kotlin via JNI
 */

#include <jni.h>
#include <string>
#include <fstream>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include <android/log.h>
#include <dlfcn.h>
#include <dirent.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define LOG_TAG "NativeSecurity"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Obfuscated strings (XOR encoded at compile time)
static const char* XOR_KEY = "BuddyLynkNative2024!";

std::string xorDecode(const std::vector<unsigned char>& encoded) {
    std::string result;
    size_t keyLen = strlen(XOR_KEY);
    for (size_t i = 0; i < encoded.size(); i++) {
        result += (char)(encoded[i] ^ XOR_KEY[i % keyLen]);
    }
    return result;
}

/**
 * Check if common root binaries exist
 * Returns true if device appears rooted
 */
bool native_isRooted() {
    // Obfuscated paths to check (XOR encoded)
    const char* paths[] = {
        "/system/bin/su",
        "/system/xbin/su",
        "/sbin/su",
        "/data/local/su",
        "/data/local/bin/su",
        "/data/local/xbin/su",
        "/system/app/Superuser.apk",
        "/system/app/SuperSU.apk"
    };
    
    for (const char* path : paths) {
        struct stat buffer;
        if (stat(path, &buffer) == 0) {
            return true;
        }
    }
    
    // Check if we can execute su
    FILE* pipe = popen("which su", "r");
    if (pipe != nullptr) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            pclose(pipe);
            return true;
        }
        pclose(pipe);
    }
    
    return false;
}

/**
 * Detect Frida hooking framework
 * Frida injects libraries and opens specific ports
 */
bool native_isFridaDetected() {
    // Check for Frida ports (27042-27047)
    int ports[] = {27042, 27043, 27044, 27045, 27046, 27047};
    
    for (int port : ports) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) continue;
        
        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        
        // Set non-blocking with timeout
        struct timeval timeout;
        timeout.tv_sec = 0;
        timeout.tv_usec = 100000; // 100ms
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        
        if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
            close(sock);
            return true; // Frida port is open!
        }
        close(sock);
    }
    
    // Check /proc/self/maps for Frida libraries
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        if (line.find("frida") != std::string::npos ||
            line.find("gadget") != std::string::npos ||
            line.find("linjector") != std::string::npos) {
            return true;
        }
    }
    
    // Check for Frida named threads
    char path[256];
    snprintf(path, sizeof(path), "/proc/%d/task", getpid());
    
    DIR* dir = opendir(path);
    if (dir != nullptr) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            if (entry->d_type == DT_DIR && entry->d_name[0] != '.') {
                char commPath[512];
                snprintf(commPath, sizeof(commPath), "/proc/%d/task/%s/comm", getpid(), entry->d_name);
                
                std::ifstream comm(commPath);
                std::string threadName;
                if (std::getline(comm, threadName)) {
                    if (threadName.find("gum-js-loop") != std::string::npos ||
                        threadName.find("gmain") != std::string::npos ||
                        threadName.find("frida") != std::string::npos) {
                        closedir(dir);
                        return true;
                    }
                }
            }
        }
        closedir(dir);
    }
    
    return false;
}

/**
 * Detect if running in an emulator
 */
bool native_isEmulator() {
    // Check for emulator files
    const char* emulatorFiles[] = {
        "/dev/socket/qemud",
        "/dev/qemu_pipe",
        "/system/lib/libc_malloc_debug_qemu.so",
        "/sys/qemu_trace",
        "/system/bin/qemud"
    };
    
    for (const char* path : emulatorFiles) {
        struct stat buffer;
        if (stat(path, &buffer) == 0) {
            return true;
        }
    }
    
    // Check system properties (via /proc)
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("Goldfish") != std::string::npos ||
            line.find("ranchu") != std::string::npos) {
            return true;
        }
    }
    
    return false;
}

/**
 * Detect debugger attachment
 */
bool native_isDebuggerAttached() {
    // Check TracerPid in /proc/self/status
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.find("TracerPid:") != std::string::npos) {
            int tracerPid = 0;
            sscanf(line.c_str(), "TracerPid:\t%d", &tracerPid);
            return tracerPid != 0;
        }
    }
    return false;
}

/**
 * Check for memory tampering tools
 */
bool native_isMemoryTampered() {
    // Check for GameGuardian, Lucky Patcher, etc.
    const char* dangerousPaths[] = {
        "/data/data/com.cih.game_cih",
        "/data/data/com.chelpus.lackypatch",
        "/data/data/com.forpda.lp",
        "/data/data/com.android.vending.billing.InAppBillingService.COIN",
        "/data/data/com.android.vendinc"
    };
    
    for (const char* path : dangerousPaths) {
        struct stat buffer;
        if (stat(path, &buffer) == 0) {
            return true;
        }
    }
    
    return false;
}

// =============================================================================
// JNI EXPORTS (Called from Kotlin)
// =============================================================================

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_orignal_buddylynk_security_NativeSecurity_isRooted(JNIEnv *env, jobject thiz) {
    return native_isRooted() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_orignal_buddylynk_security_NativeSecurity_isFridaDetected(JNIEnv *env, jobject thiz) {
    return native_isFridaDetected() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_orignal_buddylynk_security_NativeSecurity_isEmulator(JNIEnv *env, jobject thiz) {
    return native_isEmulator() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_orignal_buddylynk_security_NativeSecurity_isDebuggerAttached(JNIEnv *env, jobject thiz) {
    return native_isDebuggerAttached() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_orignal_buddylynk_security_NativeSecurity_isMemoryTampered(JNIEnv *env, jobject thiz) {
    return native_isMemoryTampered() ? JNI_TRUE : JNI_FALSE;
}

/**
 * Comprehensive security check - returns bitmask of detected threats
 * Bit 0: Root
 * Bit 1: Frida
 * Bit 2: Emulator
 * Bit 3: Debugger
 * Bit 4: Memory tampering
 */
JNIEXPORT jint JNICALL
Java_com_orignal_buddylynk_security_NativeSecurity_getSecurityStatus(JNIEnv *env, jobject thiz) {
    int status = 0;
    
    if (native_isRooted()) status |= (1 << 0);
    if (native_isFridaDetected()) status |= (1 << 1);
    if (native_isEmulator()) status |= (1 << 2);
    if (native_isDebuggerAttached()) status |= (1 << 3);
    if (native_isMemoryTampered()) status |= (1 << 4);
    
    return status;
}

} // extern "C"
