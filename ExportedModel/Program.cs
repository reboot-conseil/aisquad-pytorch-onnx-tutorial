using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

Tensor<float> PreprocessImage(string imagePath)
{
    using (Image<Rgb24> image = Image.Load<Rgb24>(imagePath))
    {
        image.Mutate(x => x.Resize(28, 28));
        var imageData = new float[3 * 28 * 28];
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                Span<Rgb24> pixelRow = accessor.GetRowSpan(y);
                for (int x = 0; x < accessor.Width; x++)
                {
                    Rgb24 pixel = pixelRow[x];
                    imageData[(0 * 28 + y) * 28 + x] = (pixelRow[x].R / 255.0f - 0.485f) / 0.229f;
                    imageData[(1 * 28 + y) * 28 + x] = (pixelRow[x].G / 255.0f - 0.456f) / 0.224f;
                    imageData[(2 * 28 + y) * 28 + x] = (pixelRow[x].B / 255.0f - 0.406f) / 0.225f;
                }
            }
        });
        Tensor<float> inputTensor = new DenseTensor<float>(imageData, new[] { 1, 3, 28, 28 });
        return inputTensor;
    }
}

string imagePath = "example_mnist_6_digit.png";
Tensor<float> inputTensor = PreprocessImage(imagePath);

InferenceSession LoadModel(string modelPath)
{
    var session = new InferenceSession(modelPath);
    return session;
}

int RunInference(InferenceSession session, Tensor<float> inputTensor)
{
    var inputs = new List<NamedOnnxValue>
    {
        NamedOnnxValue.CreateFromTensor("input_image", inputTensor)
    };
    using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs))
    {
        Tensor<float> outputTensor = results.FirstOrDefault().AsTensor<float>();
        float[] output = outputTensor.ToArray();
        int predictedDigit = output.ToList().IndexOf(output.Max());
        return predictedDigit;
    }
}


string onnxModelPath = "exported/mnist_model.onnx";
InferenceSession session = LoadModel(onnxModelPath);
int predictedDigit = RunInference(session, inputTensor);
Console.WriteLine($"Predicted Digit: {predictedDigit}");