import { useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Card } from './ui/card'
import { Button } from './ui/button'
import { Progress } from './ui/progress'

export function ModelUploadPage() {
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [modelToken, setModelToken] = useState('')

  const { getRootProps, getInputProps } = useDropzone({
    accept: {
      'application/octet-stream': ['.pt', '.pth', '.pb', '.onnx']
    },
    onDrop: async (files) => {
      if (files.length === 0) return
      
      setUploading(true)
      setProgress(0)
      
      try {
        const formData = new FormData()
        formData.append('model_file', files[0])
        
        const response = await fetch('/api/upload-model', {
          method: 'POST',
          body: formData
        })
        
        if (!response.ok) throw new Error('Upload failed')
        
        const data = await response.json()
        setModelToken(data.access_token)
      } catch (error) {
        console.error('Upload error:', error)
      } finally {
        setUploading(false)
      }
    }
  })

  return (
    <Card className="p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Upload AI Model</h1>
      
      <div
        {...getRootProps()}
        className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-gray-400"
      >
        <input {...getInputProps()} />
        <p>Drag and drop your model file here, or click to select file</p>
        <p className="text-sm text-gray-500 mt-2">
          Supports PyTorch (.pt, .pth), TensorFlow (.pb), and ONNX (.onnx) files
        </p>
      </div>

      {uploading && (
        <div className="mt-4">
          <Progress value={progress} />
          <p className="text-sm text-gray-500 mt-2">
            Uploading and optimizing model... {progress}%
          </p>
        </div>
      )}

      {modelToken && (
        <div className="mt-4 p-4 bg-green-50 rounded">
          <p className="font-medium">Model Token:</p>
          <code className="block p-2 bg-white mt-1 rounded">
            {modelToken}
          </code>
        </div>
      )}
    </Card>
  )
}

