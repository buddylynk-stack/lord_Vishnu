package com.orignal.buddylynk.ui.components

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.orignal.buddylynk.ui.utils.WindowSizeClass
import com.orignal.buddylynk.ui.utils.rememberScreenInfo

/**
 * Responsive Container - Centers content on larger screens
 */
@Composable
fun ResponsiveContainer(
    modifier: Modifier = Modifier,
    maxWidth: Dp = 600.dp,
    content: @Composable () -> Unit
) {
    val screenInfo = rememberScreenInfo()
    
    Box(
        modifier = modifier.fillMaxWidth(),
        contentAlignment = Alignment.TopCenter
    ) {
        Box(
            modifier = Modifier
                .widthIn(max = if (screenInfo.windowSizeClass == WindowSizeClass.EXPANDED) maxWidth else Dp.Infinity)
                .fillMaxWidth()
        ) {
            content()
        }
    }
}

/**
 * Responsive Row - Adjusts spacing based on screen size
 */
@Composable
fun ResponsiveRow(
    modifier: Modifier = Modifier,
    horizontalArrangement: Arrangement.Horizontal = Arrangement.Start,
    verticalAlignment: Alignment.Vertical = Alignment.Top,
    content: @Composable RowScope.() -> Unit
) {
    val screenInfo = rememberScreenInfo()
    val spacing = screenInfo.dimensions.itemSpacing
    
    Row(
        modifier = modifier,
        horizontalArrangement = when (horizontalArrangement) {
            Arrangement.SpaceBetween -> Arrangement.SpaceBetween
            Arrangement.SpaceEvenly -> Arrangement.SpaceEvenly
            Arrangement.SpaceAround -> Arrangement.SpaceAround
            else -> Arrangement.spacedBy(spacing)
        },
        verticalAlignment = verticalAlignment,
        content = content
    )
}

/**
 * Responsive Column - Adjusts spacing based on screen size
 */
@Composable
fun ResponsiveColumn(
    modifier: Modifier = Modifier,
    verticalArrangement: Arrangement.Vertical = Arrangement.Top,
    horizontalAlignment: Alignment.Horizontal = Alignment.Start,
    content: @Composable ColumnScope.() -> Unit
) {
    val screenInfo = rememberScreenInfo()
    val spacing = screenInfo.dimensions.itemSpacing
    
    Column(
        modifier = modifier,
        verticalArrangement = when (verticalArrangement) {
            Arrangement.SpaceBetween -> Arrangement.SpaceBetween
            Arrangement.SpaceEvenly -> Arrangement.SpaceEvenly
            Arrangement.SpaceAround -> Arrangement.SpaceAround
            else -> Arrangement.spacedBy(spacing)
        },
        horizontalAlignment = horizontalAlignment,
        content = content
    )
}

/**
 * Responsive Grid - Adjusts columns based on screen size
 */
@Composable
fun <T> ResponsiveGrid(
    items: List<T>,
    modifier: Modifier = Modifier,
    minColumnWidth: Dp = 150.dp,
    content: @Composable (T) -> Unit
) {
    val screenInfo = rememberScreenInfo()
    val columns = screenInfo.dimensions.gridColumns
    
    LazyVerticalGrid(
        columns = GridCells.Adaptive(minColumnWidth),
        modifier = modifier,
        contentPadding = PaddingValues(screenInfo.dimensions.screenPadding),
        horizontalArrangement = Arrangement.spacedBy(screenInfo.dimensions.itemSpacing),
        verticalArrangement = Arrangement.spacedBy(screenInfo.dimensions.itemSpacing)
    ) {
        items(items) { item ->
            content(item)
        }
    }
}

/**
 * Responsive Horizontal List - Adjusts item sizes based on screen
 */
@Composable
fun <T> ResponsiveHorizontalList(
    items: List<T>,
    modifier: Modifier = Modifier,
    contentPadding: PaddingValues = PaddingValues(0.dp),
    content: @Composable (T) -> Unit
) {
    val screenInfo = rememberScreenInfo()
    
    LazyRow(
        modifier = modifier,
        contentPadding = contentPadding,
        horizontalArrangement = Arrangement.spacedBy(screenInfo.dimensions.itemSpacing)
    ) {
        items(items) { item ->
            content(item)
        }
    }
}

/**
 * Adaptive Padding - Returns screen-appropriate padding
 */
@Composable
fun adaptiveScreenPadding(): PaddingValues {
    val screenInfo = rememberScreenInfo()
    return PaddingValues(screenInfo.dimensions.screenPadding)
}

/**
 * Adaptive Content Padding - For scrollable content with bottom nav
 */
@Composable
fun adaptiveContentPadding(): PaddingValues {
    val screenInfo = rememberScreenInfo()
    return PaddingValues(
        start = screenInfo.dimensions.screenPadding,
        end = screenInfo.dimensions.screenPadding,
        top = screenInfo.dimensions.screenPadding,
        bottom = screenInfo.dimensions.bottomNavHeight + screenInfo.dimensions.screenPadding
    )
}

/**
 * Two Pane Layout for tablets
 */
@Composable
fun TwoPaneLayout(
    modifier: Modifier = Modifier,
    mainContent: @Composable (Modifier) -> Unit,
    detailContent: @Composable (Modifier) -> Unit
) {
    val screenInfo = rememberScreenInfo()
    
    if (screenInfo.windowSizeClass == WindowSizeClass.EXPANDED) {
        // Side by side on tablets
        Row(modifier = modifier.fillMaxSize()) {
            mainContent(
                Modifier
                    .weight(0.4f)
                    .fillMaxHeight()
            )
            detailContent(
                Modifier
                    .weight(0.6f)
                    .fillMaxHeight()
            )
        }
    } else {
        // Stack on phones
        Box(modifier = modifier.fillMaxSize()) {
            mainContent(Modifier.fillMaxSize())
        }
    }
}
