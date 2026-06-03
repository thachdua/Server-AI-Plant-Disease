---
name: ios-ui-engineering
description: Builds production-quality Native iOS UIs for the Plant Disease Detection App. Must support iOS 16 (iPhone 8) and follow the specific system design for farmers (large fonts, clear icons, robust error handling). Use MVVM and SwiftUI.
---

# iOS UI Engineering (Plant Disease App - iOS 16+)

## Overview

Build production-quality iOS user interfaces tailored for the Plant Disease Detection App. The target audience includes farmers, so the UI MUST be highly intuitive, minimal, with large typography and clear iconography. The app must run flawlessly on iOS 16 (e.g., iPhone 8) and handle network latency, AI inference states, and offline edge cases gracefully.

## When to Use

- Building new SwiftUI Views (Scanner, History, Disease Dictionary, Map, Weather, Expert Q&A).
- Implementing responsive layouts for iPhone 8 up to the latest Pro Max.
- Adding MVVM state management compatible with iOS 16 (NO `@Observable` macro).
- Handling camera permissions and image picking.

## Architecture (MVVM Pattern)

### File Structure

Group files by Feature/Screen as defined in the system design:

```text
Features/
  Diagnosis/
    ScannerView.swift             # Camera & Gallery picker
    ScannerViewModel.swift        # Handles image compression & FastAPI/CoreML calls
  History/
    HistoryListView.swift         # Fetches from SQLite/CoreData
  Dictionary/                     # "Cẩm nang bệnh"
  ExpertSupport/                  # "Hỏi đáp chuyên gia"
  WeatherMap/                     # "Bản đồ dịch bệnh & Thời tiết"
```

### Component Patterns

**Separate business/AI logic from presentation (iOS 16 Standard):**

```swift
// ViewModel (State & Logic) - MUST use ObservableObject for iOS 16
@MainActor
final class ScannerViewModel: ObservableObject {
    @Published var isAnalyzing = false
    @Published var scanResult: DiagnosisResult?
    @Published var errorMessage: String?
    
    func analyzeImage(_ image: UIImage) async {
        isAnalyzing = true
        errorMessage = nil
        do {
            // Call AI Server (FastAPI) or Local CoreML
            self.scanResult = try await AIService.shared.predict(image)
        } catch {
            self.errorMessage = "Lỗi kết nối mạng, vui lòng kiểm tra lại internet."
        }
        isAnalyzing = false
    }
}

// View (Presentation only)
struct ScannerView: View {
    @StateObject private var viewModel = ScannerViewModel()
    
    var body: some View {
        ZStack {
            CameraPreview()
            
            if viewModel.isAnalyzing {
                ProgressView("Đang chẩn đoán bệnh...")
                    .padding()
                    .background(.regularMaterial)
                    .cornerRadius(12)
            }
        }
        .alert("Thông báo", isPresented: Binding(
            get: { viewModel.errorMessage != nil },
            set: { _ in viewModel.errorMessage = nil }
        )) {
            Button("OK", role: .cancel) { viewModel.errorMessage = nil }
        } message: {
            Text(viewModel.errorMessage ?? "")
        }
    }
}
```

## State Management (iOS 16 Constraints)

**CRITICAL: Do NOT use iOS 17 `@Observable`. Use the iOS 16 wrappers:**

- `@State` → Local, simple UI state.
- `@Binding` → Two-way connection to a parent's `@State`.
- `@StateObject` → Instantiating a ViewModel (Source of truth).
- `@ObservedObject` → Passing an existing ViewModel to a child view.

## Design System & Farmer-Centric UI

The target users are farmers. The UI must prioritize readability and ease of use over complex, dense layouts.

| AI Default | Plant Disease App Requirement |
|---|---|
| Tiny, dense text | Use `.font(.title3)` or `.font(.headline)` as base body text. Ensure Dynamic Type support. |
| Complex navigation | Use simple `NavigationStack` with clear back buttons and standard `TabView`. |
| Unclear error messages | Explain exactly what went wrong (e.g., "Mất mạng, không thể gửi ảnh", "Ảnh quá mờ, vui lòng chụp lại"). |
| iOS 17 `containerRelativeFrame` | Use `GeometryReader` carefully or `.frame(maxWidth: .infinity)` for iOS 16 compatibility. |

## Meaningful Empty and Error States (iOS 16 Compatible)

**CRITICAL: Do NOT use `ContentUnavailableView` (requires iOS 17). Build custom empty states:**

```swift
// iOS 16 Compatible Empty State
struct CustomEmptyView: View {
    let title: String
    let message: String
    let iconName: String
    
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: iconName)
                .font(.system(size: 48))
                .foregroundColor(.secondary)
            Text(title)
                .font(.title2)
                .fontWeight(.semibold)
            Text(message)
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// Usage in History
if history.isEmpty {
    CustomEmptyView(
        title: "Chưa có lịch sử", 
        message: "Bạn chưa có ca chẩn đoán nào. Hãy chụp ảnh lá cây để bắt đầu.", 
        iconName: "leaf"
    )
}
```

## Loading and Transitions

Provide fluid feedback when waiting for the FastAPI server or CoreML model.

```swift
// Skeleton loading for Dictionary or History lists
struct SkeletonRow: View {
    var body: some View {
        HStack {
            RoundedRectangle(cornerRadius: 8)
                .frame(width: 60, height: 60)
            VStack(alignment: .leading, spacing: 8) {
                RoundedRectangle(cornerRadius: 4)
                    .frame(height: 16)
                RoundedRectangle(cornerRadius: 4)
                    .frame(height: 12)
                    .padding(.trailing, 40)
            }
        }
        .redacted(reason: .placeholder)
    }
}
```

## Verification

After building UI for this project:
- [ ] Compiles successfully with the iOS 16.0 deployment target.
- [ ] No `ContentUnavailableView` or `@Observable` used.
- [ ] Fonts and icons are large, clear, and high-contrast (suitable for outdoor/farm use).
- [ ] Proper loading states are shown during API calls.
- [ ] Error messages are localized in Vietnamese and easy to understand.