package com.orignal.buddylynk.data.aws

/**
 * STUB: AwsConfig - Placeholder for migration to API-only
 * 
 * Media uploads should now go through the backend API's pre-signed URL endpoint.
 * This file exists only to satisfy imports during migration.
 */
object AwsConfig {
    const val REGION = "us-east-1"
    const val S3_BUCKET = "buddylynk-media-bucket-2024"
    const val POSTS_TABLE = "Buddylynk_Posts"
    const val USERS_TABLE = "Buddylynk_Users"
    const val GROUPS_TABLE = "Buddylynk_Groups"
    const val MESSAGES_TABLE = "Buddylynk_Messages"
    
    // Deprecated - do not use these directly
    const val ACCESS_KEY = ""
    const val SECRET_KEY = ""
}
