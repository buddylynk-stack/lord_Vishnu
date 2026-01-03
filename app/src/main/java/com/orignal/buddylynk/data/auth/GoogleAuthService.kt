package com.orignal.buddylynk.data.auth

import android.content.Context
import android.util.Log
import androidx.credentials.CredentialManager
import androidx.credentials.CustomCredential
import androidx.credentials.GetCredentialRequest
import androidx.credentials.GetCredentialResponse
import androidx.credentials.exceptions.GetCredentialException
import com.google.android.libraries.identity.googleid.GetGoogleIdOption
import com.google.android.libraries.identity.googleid.GoogleIdTokenCredential
import com.google.android.libraries.identity.googleid.GoogleIdTokenParsingException
import com.orignal.buddylynk.BuildConfig
import com.orignal.buddylynk.data.api.ApiService
import com.orignal.buddylynk.data.model.User
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Google Sign-In Service for BuddyLynk
 * Uses secure backend API for authentication - NO direct DynamoDB access!
 */
object GoogleAuthService {
    
    // Web Client ID from BuildConfig (set in local.properties or CI/CD)
    private val WEB_CLIENT_ID: String
        get() = BuildConfig.GOOGLE_WEB_CLIENT_ID.ifEmpty { 
            // Fallback for development only - remove in production
            "501420172548-e6frb0kf2rn4c5jqoe4g1qhuatvksadq.apps.googleusercontent.com"
        }
    
    private const val TAG = "GoogleAuthService"
    
    /**
     * Data class for Google Sign-In result
     */
    data class GoogleSignInResult(
        val success: Boolean,
        val user: User? = null,
        val errorMessage: String? = null,
        val isNewUser: Boolean = false
    )
    
    /**
     * Sign in with Google using Credential Manager
     */
    suspend fun signInWithGoogle(context: Context): GoogleSignInResult = withContext(Dispatchers.Main) {
        try {
            val credentialManager = CredentialManager.create(context)
            
            // Configure Google ID option
            val googleIdOption = GetGoogleIdOption.Builder()
                .setFilterByAuthorizedAccounts(false) // Show all accounts
                .setServerClientId(WEB_CLIENT_ID)
                .setAutoSelectEnabled(false) // Let user choose account
                .build()
            
            // Build the request
            val request = GetCredentialRequest.Builder()
                .addCredentialOption(googleIdOption)
                .build()
            
            // Get credential - must be called from Main thread with Activity context
            val result = credentialManager.getCredential(
                request = request,
                context = context
            )
            
            // Handle the result
            handleSignInResult(result)
            
        } catch (e: GetCredentialException) {
            Log.e(TAG, "Google Sign-In failed: ${e.message}", e)
            GoogleSignInResult(
                success = false,
                errorMessage = when {
                    e.message?.contains("canceled", ignoreCase = true) == true -> "Sign-in cancelled"
                    e.message?.contains("No credentials", ignoreCase = true) == true -> "No Google accounts found. Please add a Google account to your device."
                    e.message?.contains("no viable", ignoreCase = true) == true -> "No Google accounts found. Please add a Google account to your device."
                    else -> "Sign-in failed: ${e.message?.take(100)}"
                }
            )
        } catch (e: Exception) {
            Log.e(TAG, "Unexpected error during Google Sign-In", e)
            GoogleSignInResult(
                success = false,
                errorMessage = "An unexpected error occurred: ${e.message?.take(50)}"
            )
        }
    }
    
    /**
     * Handle the credential response - uses backend API for authentication
     */
    private suspend fun handleSignInResult(result: GetCredentialResponse): GoogleSignInResult = withContext(Dispatchers.IO) {
        val credential = result.credential
        
        when (credential) {
            is CustomCredential -> {
                if (credential.type == GoogleIdTokenCredential.TYPE_GOOGLE_ID_TOKEN_CREDENTIAL) {
                    try {
                        val googleIdTokenCredential = GoogleIdTokenCredential.createFrom(credential.data)
                        
                        val email = googleIdTokenCredential.id
                        val displayName = googleIdTokenCredential.displayName ?: email.substringBefore("@")
                        val profilePictureUri = googleIdTokenCredential.profilePictureUri?.toString()
                        
                        Log.d(TAG, "Google Sign-In successful: $email, $displayName")
                        
                        // Call backend API to authenticate/register user and get JWT token
                        val apiResult = ApiService.googleAuth(email, displayName, profilePictureUri)
                        
                        apiResult.fold(
                            onSuccess = { json ->
                                val userJson = json.optJSONObject("user")
                                val token = json.optString("token", "")
                                val isNewUser = json.optBoolean("isNewUser", false)
                                
                                if (userJson != null && token.isNotEmpty()) {
                                    val user = User(
                                        userId = userJson.optString("userId", ""),
                                        username = userJson.optString("username", ""),
                                        email = userJson.optString("email", ""),
                                        avatar = userJson.optString("avatar", null),
                                        avatarColor = userJson.optString("avatarColor", null),
                                        banner = userJson.optString("banner", null),
                                        bio = userJson.optString("bio", null),
                                        followersCount = userJson.optInt("followerCount", 0),
                                        followingCount = userJson.optInt("followingCount", 0)
                                    )
                                    
                                    Log.d(TAG, "Google auth API success, logging in with JWT token")
                                    
                                    // Login with JWT token - this sets ApiService.authToken
                                    AuthManager.login(user, token)
                                    
                                    GoogleSignInResult(
                                        success = true,
                                        user = user,
                                        isNewUser = isNewUser
                                    )
                                } else {
                                    Log.e(TAG, "Invalid response from Google auth API")
                                    GoogleSignInResult(
                                        success = false,
                                        errorMessage = "Invalid server response"
                                    )
                                }
                            },
                            onFailure = { e ->
                                Log.e(TAG, "Google auth API failed: ${e.message}")
                                GoogleSignInResult(
                                    success = false,
                                    errorMessage = "Authentication failed: ${e.message}"
                                )
                            }
                        )
                        
                    } catch (e: GoogleIdTokenParsingException) {
                        Log.e(TAG, "Failed to parse Google ID token", e)
                        GoogleSignInResult(
                            success = false,
                            errorMessage = "Failed to parse Google credentials"
                        )
                    }
                } else {
                    GoogleSignInResult(
                        success = false,
                        errorMessage = "Unexpected credential type"
                    )
                }
            }
            else -> {
                GoogleSignInResult(
                    success = false,
                    errorMessage = "Unexpected credential type"
                )
            }
        }
    }
}
