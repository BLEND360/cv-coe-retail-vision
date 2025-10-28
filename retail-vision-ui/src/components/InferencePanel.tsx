import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { 
  Box, 
  Typography, 
  CircularProgress, 
  Alert, 
  Card, 
  CardContent,
  Switch,
  FormControlLabel,
  Chip,
  Grid,
  Paper
} from '@mui/material';
import { Psychology, Mouse } from '@mui/icons-material';

interface Detection {
  id: number;
  class: number;
  class_name: string;
  confidence: number;
  bbox: number[];
  mask?: number[][];
}

interface InferenceResult {
  timestamp: number;
  video_time: number;
  clicked_pixel: { x: number; y: number };
  detections: Detection[];
  frame_base64: string;
  annotated_frame_base64: string;
  clicked_object: Detection | null;
  inference_type: string;
  text_prompt_used?: string;
}

interface InferencePanelProps {
  lastClickData?: { x: number; y: number; currentTime: number; frameWidth: number; frameHeight: number } | null;
  onAddToCart: (detection: { id: number; class_name: string; confidence: number }) => void;
}

type InferenceType = 'yolo-e';

const InferencePanel: React.FC<InferencePanelProps> = ({ lastClickData, onAddToCart }) => {
  const [inferenceData, setInferenceData] = useState<InferenceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAnnotated, setShowAnnotated] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [inferenceType] = useState<InferenceType>('yolo-e');
  const [textPrompt] = useState<string>('laptop, headphones, glasses, blazer, desk, watch, monitor, trash can, chair, shirt, running pants, running shoes, jacket, gloves');
  const addedDetectionsRef = useRef<Set<string>>(new Set());
  const lastInferenceTimestampRef = useRef<number>(0);

  // YOLO11 inference type selection removed - using only YOLO-E


  const fetchClickInference = useCallback(async (clickData: { x: number; y: number; currentTime: number; frameWidth: number; frameHeight: number }) => {
    try {
      setLoading(true);
      setError(null);
      
      // Use YOLO-E endpoint only
      const endpoint = '/api/inference/yolo-e';
      
      // Prepare request body
      const requestBody: any = {
        video_time: clickData.currentTime,
        x: clickData.x,
        y: clickData.y,
        frame_width: clickData.frameWidth,
        frame_height: clickData.frameHeight
      };

      // Add text prompt for YOLO-E
      if (inferenceType === 'yolo-e' && textPrompt.trim()) {
        requestBody.text_prompt = textPrompt;
      }

      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data: InferenceResult = await response.json();
      setInferenceData(data);
      setLastUpdate(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch inference data');
    } finally {
      setLoading(false);
    }
  }, [inferenceType, textPrompt]);

  // Listen for video clicks from the parent component
  useEffect(() => {
    if (lastClickData) {
      fetchClickInference(lastClickData);
    }
  }, [lastClickData, fetchClickInference]);

  // Filter detections by confidence threshold (80%)
  const filteredDetections = useMemo(() => {
    return inferenceData?.detections.filter(
      detection => detection.confidence >= 0.8
    ) || [];
  }, [inferenceData?.detections]);

  // Automatically add only the clicked object to cart
  useEffect(() => {
    if (inferenceData?.clicked_object) {
      // Check if this is a new inference session
      const currentTimestamp = inferenceData.timestamp;
      if (currentTimestamp !== lastInferenceTimestampRef.current) {
        // Reset added detections for new inference session
        addedDetectionsRef.current.clear();
        lastInferenceTimestampRef.current = currentTimestamp;
      }
      
      const detection = inferenceData.clicked_object;
      const detectionKey = `${detection.id}-${detection.class_name}-${detection.confidence.toFixed(3)}`;
      
      // Only add if not already added in this session
      if (!addedDetectionsRef.current.has(detectionKey)) {
        onAddToCart(detection);
        addedDetectionsRef.current.add(detectionKey);
      }
    }
  }, [inferenceData, onAddToCart]);

  const getCurrentFrame = () => {
    if (!inferenceData) return null;
    return showAnnotated ? inferenceData.annotated_frame_base64 : inferenceData.frame_base64;
  };


  if (loading && !inferenceData) {
    return (
      <Box sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        p: 4
      }}>
        <CircularProgress 
          size={48} 
          sx={{ 
            mb: 3,
            color: 'primary.main'
          }} 
        />
        <Typography 
          variant="h6" 
          color="text.secondary"
          sx={{ 
            fontWeight: 500,
            textAlign: 'center'
          }}
        >
          Running YOLO-E inference...
        </Typography>
      </Box>
    );
  }

  if (error && !inferenceData) {
    return (
      <Box sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        p: 4
      }}>
        <Psychology sx={{ 
          fontSize: 64, 
          color: 'error.main', 
          mb: 3, 
          opacity: 0.6 
        }} />
        <Alert 
          severity="error" 
          sx={{ 
            mb: 3,
            borderRadius: 2,
            '& .MuiAlert-message': {
              fontWeight: 500
            }
          }}
        >
          {error}
        </Alert>
        <Typography 
          variant="body2" 
          color="text.secondary" 
          textAlign="center"
          sx={{ fontWeight: 400 }}
        >
          Make sure the backend server is running on port 8000
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ 
      height: '100%', 
      display: 'flex', 
      flexDirection: 'column',
    }}>
      {/* Header */}
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography 
          variant="h5" 
          component="h2" 
          sx={{ 
            fontWeight: 600,
            color: 'text.primary'
          }}
        >
          Inference Panel
        </Typography>
        <FormControlLabel
          control={
            <Switch
              checked={showAnnotated}
              onChange={(e) => setShowAnnotated(e.target.checked)}
              size="small"
            />
          }
          label="Show Annotated"
          labelPlacement="start"
          sx={{
            '& .MuiFormControlLabel-label': {
              fontWeight: 500,
              fontSize: '0.875rem'
            }
          }}
        />
      </Box>



      {/* Instructions */}
      <Paper sx={{ 
        p: 3, 
        mb: 3, 
        bgcolor: 'grey.50',
        border: '1px solid',
        borderColor: 'grey.200',
        borderRadius: 2
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Mouse sx={{ 
            fontSize: 24, 
            color: 'text.secondary' 
          }} />
          <Typography 
            variant="body2" 
            color="text.secondary"
            sx={{ 
              fontWeight: 500,
              lineHeight: 1.5
            }}
          >
            Click anywhere on the video to run YOLO-E inference. The clicked object will be automatically added to your cart.
          </Typography>
        </Box>
      </Paper>



      {/* Frame Display */}
      {inferenceData && (
        <Box sx={{ mb: 3, textAlign: 'center' }}>
          <img
            src={`data:image/jpeg;base64,${getCurrentFrame()}`}
            alt={showAnnotated ? "Annotated Frame" : "Original Frame"}
            style={{
              maxWidth: '100%',
              maxHeight: '300px',
              borderRadius: '12px',
              border: '1px solid rgba(0, 0, 0, 0.08)',
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
            }}
          />
          <Typography 
            variant="caption" 
            color="text.secondary" 
            display="block" 
            mt={2}
            sx={{ 
              fontWeight: 500,
              fontSize: '0.75rem'
            }}
          >
            {showAnnotated ? 'Annotated' : 'Original'} Frame - {inferenceData.inference_type || inferenceType.toUpperCase()}
          </Typography>
        </Box>
      )}

      {/* Clicked Object Highlight */}
      {inferenceData?.clicked_object && (
        <Paper sx={{ 
          p: 3, 
          mb: 3, 
          bgcolor: 'grey.50',
          border: '2px solid',
          borderColor: 'primary.main',
          borderRadius: 2
        }}>
          <Typography 
            variant="h6" 
            sx={{ 
              mb: 2, 
              color: 'text.primary',
              fontWeight: 600
            }}
          >
            🎯 Object at Clicked Pixel
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Box>
              <Typography 
                variant="subtitle1" 
                sx={{ 
                  fontWeight: 600, 
                  color: 'text.primary',
                  mb: 0.5
                }}
              >
                {inferenceData.clicked_object.class_name}
              </Typography>
              <Typography 
                variant="body2" 
                color="text.secondary"
                sx={{ fontWeight: 500 }}
              >
                Confidence: {(inferenceData.clicked_object.confidence * 100).toFixed(1)}%
              </Typography>
            </Box>
            <Chip 
              label={`${(inferenceData.clicked_object.confidence * 100).toFixed(0)}%`}
              sx={{
                bgcolor: 'primary.main',
                color: 'white',
                fontWeight: 600
              }}
            />
          </Box>
        </Paper>
      )}

      {/* All Detections */}
      {filteredDetections.length > 0 && (
        <Box sx={{ 
          flex: 1, 
          overflow: 'auto',
          minHeight: 0, // Important for flex child to shrink
        }}>
          <Typography 
            variant="h6" 
            sx={{ 
              mb: 3,
              fontWeight: 600,
              color: 'text.primary'
            }}
          >
            All Detected Objects ({filteredDetections.length})
          </Typography>
          
          <Grid container spacing={2}>
            {filteredDetections.map((detection, index) => (
              <Grid item xs={12} key={index}>
                <Card 
                  variant="outlined" 
                  sx={{ 
                    borderColor: inferenceData?.clicked_object?.id === detection.id ? 'primary.main' : 'grey.200',
                    borderWidth: inferenceData?.clicked_object?.id === detection.id ? 2 : 1,
                    borderRadius: 2,
                    '&:hover': {
                      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
                    }
                  }}
                >
                  <CardContent sx={{ p: '16px !important' }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Box>
                        <Typography 
                          variant="subtitle2" 
                          sx={{ 
                            fontWeight: 600,
                            color: 'text.primary',
                            mb: 0.5
                          }}
                        >
                          {detection.class_name}
                        </Typography>
                        <Typography 
                          variant="caption" 
                          color="text.secondary"
                          sx={{ fontWeight: 500 }}
                        >
                          ID: {detection.id} | Confidence: {(detection.confidence * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                      <Chip 
                        label={`${(detection.confidence * 100).toFixed(0)}%`}
                        size="small" 
                        sx={{
                          bgcolor: detection.confidence >= 0.9 ? '#34C759' : '#FF9500',
                          color: 'white',
                          fontWeight: 600
                        }}
                      />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}


      {/* Waiting State */}
      {!inferenceData && !loading && !error && (
        <Box sx={{ 
          flex: 1, 
          display: 'flex', 
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          p: 4
        }}>
          <Mouse sx={{ 
            fontSize: 64, 
            color: 'text.secondary', 
            mb: 3, 
            opacity: 0.4 
          }} />
          <Typography 
            variant="h6" 
            color="text.secondary" 
            textAlign="center" 
            mb={2}
            sx={{ fontWeight: 600 }}
          >
            Click on Video
          </Typography>
          <Typography 
            variant="body2" 
            color="text.secondary" 
            textAlign="center"
            sx={{ 
              fontWeight: 500,
              lineHeight: 1.5,
              maxWidth: '300px'
            }}
          >
            Click anywhere on the video player to run YOLO-E inference. The clicked object will be automatically added to your cart.
          </Typography>
        </Box>
      )}

      {/* Status Footer */}
      {lastUpdate && (
        <Box sx={{ 
          mt: 'auto', 
          pt: 2, 
          borderTop: '1px solid',
          borderColor: 'grey.200'
        }}>
          <Typography 
            variant="caption" 
            color="text.secondary"
            sx={{ 
              fontWeight: 500,
              fontSize: '0.75rem'
            }}
          >
            Last inference: {lastUpdate.toLocaleTimeString()} | Method: {inferenceType.toUpperCase()}
            {inferenceData?.text_prompt_used && ` | Prompt: ${inferenceData.text_prompt_used}`}
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default InferencePanel;
