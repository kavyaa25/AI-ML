"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts"
import { Activity, Zap, Brain, Settings, TrendingUp, Clock, MemoryStick, Cpu } from "lucide-react"

export default function InteractiveAIOptimizer() {
  const [selectedModel, setSelectedModel] = useState("baseline")
  const [inputText, setInputText] = useState("")
  const [prediction, setPrediction] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [metrics, setMetrics] = useState({
    inferenceTime: 0,
    memoryUsage: 0,
    throughput: 0,
    accuracy: 0,
  })
  const [optimizationProgress, setOptimizationProgress] = useState(0)
  const [performanceHistory, setPerformanceHistory] = useState([])

  // Mock model configurations
  const models = {
    baseline: {
      name: "Baseline ResNet50",
      description: "Original unoptimized model",
      color: "#FF6B6B",
      avgTime: 45.2,
      memory: 1200,
      accuracy: 76.5,
    },
    quantized: {
      name: "Quantized Model",
      description: "INT8 quantization optimization",
      color: "#4ECDC4",
      avgTime: 28.1,
      memory: 300,
      accuracy: 75.8,
    },
    torchscript: {
      name: "TorchScript Optimized",
      description: "JIT compilation with graph optimization",
      color: "#45B7D1",
      avgTime: 32.5,
      memory: 1100,
      accuracy: 76.3,
    },
    mixed_precision: {
      name: "Mixed Precision",
      description: "FP16 optimization for modern GPUs",
      color: "#96CEB4",
      avgTime: 22.8,
      memory: 800,
      accuracy: 76.1,
    },
    pruned: {
      name: "Pruned Model",
      description: "Structured pruning optimization",
      color: "#FFEAA7",
      avgTime: 35.7,
      memory: 950,
      accuracy: 74.9,
    },
    optimized: {
      name: "Combined Optimized",
      description: "Best-of-all optimization techniques",
      color: "#DDA0DD",
      avgTime: 18.3,
      memory: 650,
      accuracy: 75.5,
    },
  }

  // Performance comparison data
  const performanceData = Object.entries(models).map(([key, model]) => ({
    name: model.name.split(" ")[0],
    time: model.avgTime,
    memory: model.memory,
    accuracy: model.accuracy,
    improvement: (((models.baseline.avgTime - model.avgTime) / models.baseline.avgTime) * 100).toFixed(1),
  }))

  // Simulate AI inference
  const runInference = async () => {
    if (!inputText.trim()) {
      setPrediction("Please enter some text to analyze")
      return
    }

    setIsLoading(true)
    setOptimizationProgress(0)

    // Simulate optimization progress
    const progressInterval = setInterval(() => {
      setOptimizationProgress((prev) => {
        if (prev >= 100) {
          clearInterval(progressInterval)
          return 100
        }
        return prev + 10
      })
    }, 100)

    // Simulate inference delay based on model
    const model = models[selectedModel]
    await new Promise((resolve) => setTimeout(resolve, model.avgTime * 20))

    // Mock prediction results
    const predictions = [
      "Positive sentiment detected with high confidence",
      "Negative sentiment with moderate confidence",
      "Neutral sentiment detected",
      "Mixed emotions - both positive and negative elements",
      "Highly positive sentiment with excitement indicators",
    ]

    const randomPrediction = predictions[Math.floor(Math.random() * predictions.length)]
    setPrediction(randomPrediction)

    // Update metrics
    setMetrics({
      inferenceTime: model.avgTime + (Math.random() - 0.5) * 5,
      memoryUsage: model.memory + Math.random() * 100,
      throughput: 1000 / model.avgTime,
      accuracy: model.accuracy + (Math.random() - 0.5) * 2,
    })

    // Add to performance history
    setPerformanceHistory((prev) => [
      ...prev.slice(-9),
      {
        time: new Date().toLocaleTimeString(),
        value: model.avgTime + (Math.random() - 0.5) * 5,
        model: model.name,
      },
    ])

    setIsLoading(false)
    setOptimizationProgress(100)
  }

  // Auto-generate sample text
  const generateSampleText = () => {
    const samples = [
      "I absolutely love this new AI optimization tool! It's incredibly fast and efficient.",
      "The performance improvements are disappointing. Expected much better results.",
      "This is a decent tool with some useful features, though it could be improved.",
      "Amazing breakthrough in AI inference optimization! Revolutionary technology.",
      "The interface is confusing and the results are inconsistent.",
    ]
    setInputText(samples[Math.floor(Math.random() * samples.length)])
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center gap-3">
            <Brain className="h-8 w-8 text-blue-600" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Interactive AI Optimizer
            </h1>
          </div>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Experience real-time AI model optimization with interactive performance analysis and comparison
          </p>
        </div>

        {/* Main Interface */}
        <Tabs defaultValue="inference" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="inference" className="flex items-center gap-2">
              <Zap className="h-4 w-4" />
              Inference
            </TabsTrigger>
            <TabsTrigger value="optimization" className="flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Optimization
            </TabsTrigger>
            <TabsTrigger value="performance" className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Performance
            </TabsTrigger>
            <TabsTrigger value="monitoring" className="flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Monitoring
            </TabsTrigger>
          </TabsList>

          {/* Inference Tab */}
          <TabsContent value="inference" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Input Section */}
              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="h-5 w-5" />
                    AI Inference Interface
                  </CardTitle>
                  <CardDescription>Enter text for sentiment analysis using optimized AI models</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Select Model:</label>
                    <Select value={selectedModel} onValueChange={setSelectedModel}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {Object.entries(models).map(([key, model]) => (
                          <SelectItem key={key} value={key}>
                            <div className="flex items-center gap-2">
                              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: model.color }} />
                              {model.name}
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <label className="text-sm font-medium">Input Text:</label>
                      <Button variant="outline" size="sm" onClick={generateSampleText}>
                        Generate Sample
                      </Button>
                    </div>
                    <Textarea
                      placeholder="Enter your text here for sentiment analysis..."
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      rows={4}
                    />
                  </div>

                  <Button onClick={runInference} disabled={isLoading} className="w-full" size="lg">
                    {isLoading ? "Processing..." : "Run AI Inference"}
                  </Button>

                  {isLoading && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Optimization Progress</span>
                        <span>{optimizationProgress}%</span>
                      </div>
                      <Progress value={optimizationProgress} />
                    </div>
                  )}

                  {prediction && (
                    <Alert>
                      <Brain className="h-4 w-4" />
                      <AlertDescription className="font-medium">Prediction: {prediction}</AlertDescription>
                    </Alert>
                  )}
                </CardContent>
              </Card>

              {/* Model Info */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Model Information</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 rounded-full" style={{ backgroundColor: models[selectedModel].color }} />
                      <span className="font-medium">{models[selectedModel].name}</span>
                    </div>
                    <p className="text-sm text-gray-600">{models[selectedModel].description}</p>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 bg-blue-50 rounded-lg">
                      <Clock className="h-5 w-5 mx-auto mb-1 text-blue-600" />
                      <div className="text-sm font-medium">{models[selectedModel].avgTime}ms</div>
                      <div className="text-xs text-gray-500">Avg Time</div>
                    </div>
                    <div className="text-center p-3 bg-green-50 rounded-lg">
                      <MemoryStick className="h-5 w-5 mx-auto mb-1 text-green-600" />
                      <div className="text-sm font-medium">{models[selectedModel].memory}MB</div>
                      <div className="text-xs text-gray-500">Memory</div>
                    </div>
                  </div>

                  <div className="text-center p-3 bg-purple-50 rounded-lg">
                    <div className="text-lg font-bold text-purple-600">{models[selectedModel].accuracy}%</div>
                    <div className="text-xs text-gray-500">Accuracy</div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Optimization Tab */}
          <TabsContent value="optimization" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Object.entries(models).map(([key, model]) => (
                <Card key={key} className="hover:shadow-lg transition-shadow">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg flex items-center gap-2">
                        <div className="w-4 h-4 rounded-full" style={{ backgroundColor: model.color }} />
                        {model.name}
                      </CardTitle>
                      {key === "optimized" && (
                        <Badge variant="secondary" className="bg-green-100 text-green-800">
                          Best
                        </Badge>
                      )}
                    </div>
                    <CardDescription>{model.description}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div className="flex items-center gap-2">
                        <Clock className="h-4 w-4 text-blue-500" />
                        <span>{model.avgTime}ms</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <MemoryStick className="h-4 w-4 text-green-500" />
                        <span>{model.memory}MB</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Cpu className="h-4 w-4 text-purple-500" />
                        <span>{model.accuracy}%</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <TrendingUp className="h-4 w-4 text-orange-500" />
                        <span>
                          {(((models.baseline.avgTime - model.avgTime) / models.baseline.avgTime) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    <Button
                      variant={selectedModel === key ? "default" : "outline"}
                      className="w-full"
                      onClick={() => setSelectedModel(key)}
                    >
                      {selectedModel === key ? "Selected" : "Select Model"}
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Performance Tab */}
          <TabsContent value="performance" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Inference Time Comparison</CardTitle>
                  <CardDescription>Lower is better</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={performanceData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="time" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Memory Usage Comparison</CardTitle>
                  <CardDescription>Memory consumption in MB</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={performanceData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="memory" fill="#82ca9d" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics Summary</CardTitle>
                <CardDescription>Comprehensive comparison of all optimization techniques</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left p-2">Model</th>
                        <th className="text-left p-2">Inference Time</th>
                        <th className="text-left p-2">Memory Usage</th>
                        <th className="text-left p-2">Accuracy</th>
                        <th className="text-left p-2">Speed Improvement</th>
                      </tr>
                    </thead>
                    <tbody>
                      {performanceData.map((row, index) => (
                        <tr key={index} className="border-b hover:bg-gray-50">
                          <td className="p-2 font-medium">{row.name}</td>
                          <td className="p-2">{row.time}ms</td>
                          <td className="p-2">{row.memory}MB</td>
                          <td className="p-2">{row.accuracy}%</td>
                          <td className="p-2">
                            <Badge variant={Number.parseFloat(row.improvement) > 0 ? "default" : "secondary"}>
                              {row.improvement}%
                            </Badge>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Monitoring Tab */}
          <TabsContent value="monitoring" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">Inference Time</p>
                      <p className="text-2xl font-bold">{metrics.inferenceTime.toFixed(1)}ms</p>
                    </div>
                    <Clock className="h-8 w-8 text-blue-500" />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">Memory Usage</p>
                      <p className="text-2xl font-bold">{metrics.memoryUsage.toFixed(0)}MB</p>
                    </div>
                    <MemoryStick className="h-8 w-8 text-green-500" />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">Throughput</p>
                      <p className="text-2xl font-bold">{metrics.throughput.toFixed(1)} FPS</p>
                    </div>
                    <TrendingUp className="h-8 w-8 text-purple-500" />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">Accuracy</p>
                      <p className="text-2xl font-bold">{metrics.accuracy.toFixed(1)}%</p>
                    </div>
                    <Cpu className="h-8 w-8 text-orange-500" />
                  </div>
                </CardContent>
              </Card>
            </div>

            {performanceHistory.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Real-time Performance Monitoring</CardTitle>
                  <CardDescription>Live inference time tracking</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={performanceHistory}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />
                      <Line
                        type="monotone"
                        dataKey="value"
                        stroke="#8884d8"
                        strokeWidth={2}
                        dot={{ fill: "#8884d8" }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>

        {/* Footer */}
        <div className="text-center text-sm text-gray-500 py-4">
          <p>🚀 Interactive AI Optimizer - Demonstrating real-time model optimization and performance analysis</p>
        </div>
      </div>
    </div>
  )
}
