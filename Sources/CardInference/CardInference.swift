import Foundation
import AVFoundation
import Vision
import CoreML

@available(iOS 17.0, *)
public class CardInference: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, ObservableObject, @unchecked Sendable {
    private var captureSession: AVCaptureSession?
    private var suitModel: VNCoreMLModel?
    private var rankModel: VNCoreMLModel?
    private let serialQueue = DispatchQueue(label: "com.cardinference.serialQueue")
    
    public var inferenceCallback: ((String, Float, String, Float) -> Void)?
    
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
            print("Failed to set up front camera")
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
    
    private func performInference(on pixelBuffer: CVPixelBuffer) {
        guard let suitModel = suitModel, let rankModel = rankModel else {
            print("Vision models not set up")
            return
        }
        
        let suitRequest = VNCoreMLRequest(model: suitModel) { [weak self] request, error in
            guard let results = request.results as? [VNClassificationObservation],
                  let topResult = results.first else { return }
            
            let suitIdentifier = topResult.identifier
            let suitConfidence = topResult.confidence
            
            self?.performRankInference(on: pixelBuffer, suitIdentifier: suitIdentifier, suitConfidence: suitConfidence)
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        do {
            try handler.perform([suitRequest])
        } catch {
            print("Failed to perform suit inference: \(error)")
        }
    }
    
    private func performRankInference(on pixelBuffer: CVPixelBuffer, suitIdentifier: String, suitConfidence: Float) {
        guard let rankModel = rankModel else {
            print("Rank model not set up")
            return
        }
        
        let rankRequest = VNCoreMLRequest(model: rankModel) { [weak self] request, error in
            guard let results = request.results as? [VNClassificationObservation],
                  let topResult = results.first else { return }
            
            let rankIdentifier = topResult.identifier
            let rankConfidence = topResult.confidence
            
            self?.serialQueue.async { [weak self] in
                self?.inferenceCallback?(suitIdentifier, suitConfidence, rankIdentifier, rankConfidence)
            }
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        do {
            try handler.perform([rankRequest])
        } catch {
            print("Failed to perform rank inference: \(error)")
        }
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
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        performInference(on: pixelBuffer)
    }
}
