using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NeuralNetwork.Models;
public class Sample
{
    public string Label { get; set; } = string.Empty;

    public Vector<double> X { get; set; } = Vector<double>.Build.Dense(0);

    public Vector<double> ExpectedResult { get; set; } = Vector<double>.Build.Dense(0);
}
