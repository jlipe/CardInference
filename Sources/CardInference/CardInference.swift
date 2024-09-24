import Foundation
import AVFoundation
import Vision
import CoreML

public enum Suit: String, CaseIterable, Sendable {
    case heart, diamond, club, spade
}

public enum Rank: String, CaseIterable, Sendable {
    case ace, two, three, four, five, six, seven, eight, nine, ten, jack, queen, king
}

public struct CardInferenceResult: Equatable, Sendable {
    internal let suit: Suit
    internal let rank: Rank
    
    public var suitString: String {
        return suit.rawValue
    }
    
    public var rankString: String {
        return rank.rawValue
    }
    
    public var cardString: String {
        return "\(rankString)\(suitString)"
    }
}

@available(iOS 17.0, *)
public class CardInference: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, ObservableObject, @unchecked Sendable {
    private var captureSession: AVCaptureSession?
    private var suitModel: VNCoreMLModel?
    private var rankModel: VNCoreMLModel?
    private let serialQueue = DispatchQueue(label: "com.cardinference.serialQueue")
    
    public var inferenceCallback: ((CardInferenceResult?) -> Void)?
    
    // Confidence thresholds
    public var suitConfidenceThreshold: Float = 0.99
    public var rankConfidenceThreshold: Float = 0.95
    
    public override init() {
        super.init()
    }
    
    public func setup() {
        serialQueue.sync {
            self.setupCamera()
            self.setupVision()
        }
    }
    
    private func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession?.sessionPreset = .medium
        
        guard let frontCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: frontCamera) else {
            return
        }
        
        captureSession?.addInput(input)
        
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: serialQueue)
        captureSession?.addOutput(videoOutput)
    }
    
    private func setupVision() {
        guard let suitModelURL = Bundle.module.url(forResource: "SuitImageClassifier", withExtension: "mlmodelc"),
              let suitModel = try? MLModel(contentsOf: suitModelURL),
              let suitVisionModel = try? VNCoreMLModel(for: suitModel),
              let rankModelURL = Bundle.module.url(forResource: "RankClassifier", withExtension: "mlmodelc"),
              let rankModel = try? MLModel(contentsOf: rankModelURL),
              let rankVisionModel = try? VNCoreMLModel(for: rankModel) else {
            print("Failed to load Core ML models")
            return
        }
        
        self.suitModel = suitVisionModel
        self.rankModel = rankVisionModel
    }
    
    public func startInference() {
        serialQueue.async { [weak self] in
            self?.captureSession?.startRunning()
        }
    }
    
    public func stopInference() {
        serialQueue.async { [weak self] in
            self?.captureSession?.stopRunning()
        }
    }
    
    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            print("Failed to get pixel buffer from sample buffer")
            return
        }
        performInference(on: pixelBuffer)
    }
    
    private func performInference(on pixelBuffer: CVPixelBuffer) {
        guard let suitModel = suitModel, let rankModel = rankModel else {
            print("Vision models not set up")
            return
        }
        
        let suitRequest = VNCoreMLRequest(model: suitModel) { [weak self] request, error in
            if let error = error {
                print("Suit inference error: \(error.localizedDescription)")
                self?.inferenceCallback?(nil)
                return
            }
            
            guard let results = request.results as? [VNClassificationObservation],
                  let topResult = results.first else {
                self?.inferenceCallback?(nil)
                return
            }
            
            guard let suit = Suit(rawValue: topResult.identifier),
                  topResult.confidence >= self!.suitConfidenceThreshold else {
                self?.inferenceCallback?(nil)
                return
            }
            
            self?.performRankInference(on: pixelBuffer, suit: suit)
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        do {
            try handler.perform([suitRequest])
        } catch {
            print("Failed to perform suit inference: \(error)")
            self.inferenceCallback?(nil)
        }
    }
    
    private func performRankInference(on pixelBuffer: CVPixelBuffer, suit: Suit) {
        guard let rankModel = rankModel else {
            print("Rank model not set up")
            self.inferenceCallback?(nil)
            return
        }
        
        let rankRequest = VNCoreMLRequest(model: rankModel) { [weak self] request, error in
            guard let self = self else { return }
            if let error = error {
                print("Rank inference error: \(error.localizedDescription)")
                self.inferenceCallback?(nil)
                return
            }
            
            guard let results = request.results as? [VNClassificationObservation],
                  let topResult = results.first else {
                self.inferenceCallback?(nil)
                return
            }
            
            guard let rank = Rank(rawValue: topResult.identifier),
                  topResult.confidence >= self.rankConfidenceThreshold else {
                self.inferenceCallback?(nil)
                return
            }
            
            let result = CardInferenceResult(suit: suit, rank: rank)
            
            self.serialQueue.async { [weak self] in
                self?.inferenceCallback?(result)
            }
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        do {
            try handler.perform([rankRequest])
        } catch {
            print("Failed to perform rank inference: \(error)")
            self.inferenceCallback?(nil)
        }
    }
}
