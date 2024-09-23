import Foundation
import AVFoundation
import Vision
import CoreML

@available(iOS 17.0, *)
public class CardInference: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, ObservableObject, @unchecked Sendable {
    private var captureSession: AVCaptureSession?
    private var visionModel: VNCoreMLModel?
    private let serialQueue = DispatchQueue(label: "com.cardinference.serialQueue")
    
    public var inferenceCallback: ((String, Float) -> Void)?
    
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
        guard let modelURL = Bundle.module.url(forResource: "SuitImageClassifier", withExtension: "mlmodelc"),
              let model = try? MLModel(contentsOf: modelURL),
              let visionModel = try? VNCoreMLModel(for: model) else {
            print("Failed to load Core ML model")
            return
        }
        
        self.visionModel = visionModel
    }
    
    private func performInference(on pixelBuffer: CVPixelBuffer) {
        guard let visionModel = visionModel else {
            print("Vision model not set up")
            return
        }
        
        let request = VNCoreMLRequest(model: visionModel) { [weak self] request, error in
            guard let results = request.results as? [VNClassificationObservation],
                  let topResult = results.first else { return }
            
            let identifier = topResult.identifier
            let confidence = topResult.confidence
            
            self?.serialQueue.async { [weak self] in
                self?.inferenceCallback?(identifier, confidence)
            }
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        do {
            try handler.perform([request])
        } catch {
            print("Failed to perform inference: \(error)")
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
