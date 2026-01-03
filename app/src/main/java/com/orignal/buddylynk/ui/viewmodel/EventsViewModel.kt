package com.orignal.buddylynk.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.orignal.buddylynk.data.auth.AuthManager
import com.orignal.buddylynk.data.aws.DynamoDbService
import com.orignal.buddylynk.data.model.Event
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class EventsViewModel : ViewModel() {

    private val _events = MutableStateFlow<List<Event>>(emptyList())
    val events: StateFlow<List<Event>> = _events.asStateFlow()

    private val _isLoading = MutableStateFlow(true)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    private val _selectedFilter = MutableStateFlow("All")
    val selectedFilter: StateFlow<String> = _selectedFilter.asStateFlow()

    init {
        loadEvents()
    }

    fun loadEvents() {
        viewModelScope.launch {
            _isLoading.value = true
            try {
                val fetched = DynamoDbService.getEvents()
                // Simple sort by date for now (string comparison isn't ideal but works for 'yyyy-MM-dd' or similar if unified)
                // Assuming fetched returns raw list.
                _events.value = fetched
            } catch (e: Exception) {
                // handle error
            } finally {
                _isLoading.value = false
            }
        }
    }

    fun setFilter(filter: String) {
        _selectedFilter.value = filter
        // filtering logic could also happen here or in UI
        if (filter == "All") {
             loadEvents()
        } else {
            // Primitive client side filter
            viewModelScope.launch {
                val all = DynamoDbService.getEvents()
                _events.value = when(filter) {
                    "Online" -> all.filter { it.isOnline }
                    // Add other filters as data structure permits
                    else -> all
                }
            }
        }
    }

    // Creating event placeholder
    fun createEvent(title: String, date: String, time: String) {
        viewModelScope.launch {
            val user = AuthManager.currentUser.value ?: return@launch
            val event = Event(
                eventId = "evt_" + System.currentTimeMillis(),
                title = title,
                description = "New user event",
                date = date,
                time = time,
                location = "Online",
                isOnline = true,
                attendeesCount = 1,
                organizerId = user.userId,
                createdAt = System.currentTimeMillis().toString(),
                category = "General"
            )
            val success = DynamoDbService.createEvent(event)
            if (success) loadEvents()
        }
    }
}
