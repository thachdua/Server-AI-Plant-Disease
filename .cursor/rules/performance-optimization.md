---
name: ios-performance-optimization
description: Optimizes iOS application performance for iOS 16 devices (e.g., iPhone 8) and low-connectivity environments. Use when writing CoreML/FastAPI inference logic, handling high-resolution camera images, managing SwiftUI state, or when the app experiences hangs or memory crashes.
---

# iOS Performance Optimization (iOS 16 & Farm Constraints)

## Overview

Measure before optimizing. For this Plant Disease App running on older devices (iPhone 8 / iOS 16) in farm environments, performance means: preventing Out-Of-Memory (OOM) crashes due to 2GB RAM limits, avoiding Main Thread freezes during AI inference, and handling poor network conditions when calling the FastAPI server.

## When to Use

- Writing CoreML local inference or FastAPI network requests.
- Handling `UIImage` from the camera or gallery.
- Dealing with long lists in the History or Dictionary tabs.
- When Xcode reports High Memory usage or the UI freezes.

## iOS 16 / iPhone 8 Targets

| Metric | Good | Needs Improvement | Poor (High Risk of Crash on iPhone 8) |
|--------|------|-------------------|------|
| **Memory Footprint**| < 100MB | 100MB - 200MB | > 200MB (OOM Danger!) |
| **API Timeout** | 10s | 15s | > 20s (User abandons app) |
| **Frame Rate** | 60 fps | 30 - 59 fps | < 30 fps (Janky) |

## The Optimization Workflow

1. **MEASURE** → Use Xcode Instruments (Time Profiler, Allocations, Network).
2. **IDENTIFY** → Find the bottleneck (e.g., image is 15MB, API takes 10s).
3. **FIX** → Downsample image, add network timeout, move to background thread.
4. **VERIFY** → Re-run Instruments.

## Step 1: Fix Common iOS/AI Anti-Patterns

### 1. Blocking the Main Thread (The AI Trap)

**Rule:** NEVER run CoreML inference or heavy image compression on the Main Thread.

```swift
// BAD: Blocks the UI.
func analyzeLeaf(image: UIImage) {
    let result = try? localModel.prediction(input: resizedImage) 
    self.diseaseResult = result
}

// GOOD: Use Swift Concurrency
func analyzeLeaf(image: UIImage) async {
    self.isAnalyzing = true
    
    // Move heavy computation to a background thread
    let result = await Task.detached(priority: .userInitiated) {
        let resizedImage = self.downsampleImage(image: image, to: CGSize(width: 224, height: 224))
        return try? self.localModel.prediction(input: resizedImage)
    }.value
    
    await MainActor.run {
        self.diseaseResult = result
        self.isAnalyzing = false
    }
}
```

### 2. Memory Spikes with High-Res Images (iPhone 8 Killer)

**Rule:** iPhone 8 has only 2GB RAM. A 12MP photo can crash the app if loaded fully into memory multiple times. ALWAYS downsample.

```swift
// GOOD: Downsample directly from Data without inflating the full image in RAM
func downsampleImage(imageData: Data, to pointSize: CGSize, scale: CGFloat = UIScreen.main.scale) -> UIImage? {
    let imageSourceOptions = [kCGImageSourceShouldCache: false] as CFDictionary
    guard let imageSource = CGImageSourceCreateWithData(imageData as CFData, imageSourceOptions) else { return nil }
    
    let maxDimensionInPixels = max(pointSize.width, pointSize.height) * scale
    let downsampleOptions = [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceShouldCacheImmediately: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: maxDimensionInPixels
    ] as CFDictionary
    
    guard let downsampledImage = CGImageSourceCreateThumbnailAtIndex(imageSource, 0, downsampleOptions) else { return nil }
    return UIImage(cgImage: downsampledImage)
}
```

### 3. Network Optimization (Farm Environment)

**Rule:** Farms often have weak 3G/4G. Never leave an API call hanging forever. Always compress images BEFORE uploading.

```swift
// GOOD: Set strict timeouts and compress images before sending to FastAPI
func uploadImageToFastAPI(image: UIImage) async throws -> DiagnosisResult {
    // 1. Compress aggressively for weak networks
    guard let jpegData = image.jpegData(compressionQuality: 0.6) else { throw URLError(.badServerResponse) }
    
    // 2. Set strict timeout
    var request = URLRequest(url: URL(string: "[https://your-fastapi.com/predict](https://your-fastapi.com/predict)")!)
    request.httpMethod = "POST"
    request.timeoutInterval = 15.0 // Don't let it hang forever
    
    // 3. Network call
    let (data, response) = try await URLSession.shared.upload(for: request, from: jpegData)
    return try JSONDecoder().decode(DiagnosisResult.self, from: data)
}
```

### 4. SwiftUI Layout (iOS 16 Constraints)

**Rule:** `GeometryReader` is expensive. Do NOT use iOS 17 `.containerRelativeFrame`.

```swift
// BAD: Expensive layout or iOS 17 only code
GeometryReader { geo in Image(uiImage: leaf).frame(width: geo.size.width) }

// GOOD: iOS 16 native scaling
Image(uiImage: leaf)
    .resizable()
    .scaledToFit()
    .frame(maxWidth: .infinity) // Safe and performant on iOS 16
    .padding()
```

## Step 2: Caching & SQLite/Core Data

- **Clean Up:** Set `image = nil` on Views when they disappear to immediately free up RAM on the iPhone 8.
- **Image Caching:** Use `NSCache` for thumbnails in the History list.
- **Core Data:** Use `.fetchBatchSize(20)` when displaying the History tab so the iPhone 8 doesn't load 1000 past scans into RAM at once.

## Verification

Before committing performance changes:
- [ ] No Main Thread warnings appear in Xcode.
- [ ] Memory footprint stays under 150MB after taking 5 consecutive photos.
- [ ] Network requests to FastAPI have a `timeoutInterval` defined.
- [ ] Code compiles on iOS 16 (No `.containerRelativeFrame` used).