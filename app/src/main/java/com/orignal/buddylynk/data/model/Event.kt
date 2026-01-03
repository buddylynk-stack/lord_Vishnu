package com.orignal.buddylynk.data.model

data class Event(
    val eventId: String,
    val title: String,
    val description: String,
    val date: String, // e.g. "Dec 15, 2025"
    val time: String, // e.g. "7:00 PM"
    val location: String,
    val isOnline: Boolean,
    val attendeesCount: Int,
    val organizerId: String,
    val createdAt: String,
    val category: String, // e.g. "Tech", "Music"
    val imageUrl: String? = null
)
