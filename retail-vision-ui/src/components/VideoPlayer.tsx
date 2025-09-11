import React, { useState, useRef, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  IconButton, 
  Slider, 
  Stack,
  Paper
} from '@mui/material';
import { 
  PlayArrow, 
  Pause, 
  VolumeUp, 
  VolumeOff,
  Fullscreen
} from '@mui/icons-material';

interface VideoPlayerProps {
  onTimeUpdate?: (currentTime: number) => void; // Callback to parent with current time
  onVideoClick?: (clickData: { x: number; y: number; currentTime: number; frameWidth: number; frameHeight: number }) => void; // Callback for video clicks
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ onTimeUpdate, onVideoClick }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const videoContainerRef = useRef<HTMLDivElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);

  const videoUrl = "/The BLEND360 Approach.mp4";

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoadedMetadata = () => {
      setDuration(video.duration);
    };

    const handleTimeUpdate = () => {
      const time = video.currentTime;
      setCurrentTime(time);
      // Notify parent component of time updates
      onTimeUpdate?.(time);
    };

    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    video.addEventListener('timeupdate', handleTimeUpdate);

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('timeupdate', handleTimeUpdate);
    };
  }, [onTimeUpdate]);

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleSeek = (event: Event, newValue: number | number[]) => {
    const time = newValue as number;
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
      onTimeUpdate?.(time);
    }
  };

  const handleVolumeChange = (event: Event, newValue: number | number[]) => {
    const newVolume = newValue as number;
    setVolume(newVolume);
    if (videoRef.current) {
      videoRef.current.volume = newVolume;
    }
  };

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const handleVideoClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!onVideoClick || !videoRef.current) return;

    const rect = event.currentTarget.getBoundingClientRect();
    const x = Math.round(event.clientX - rect.left);
    const y = Math.round(event.clientY - rect.top);
    
    // Get actual video dimensions
    const video = videoRef.current;
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
    
    // Calculate the actual displayed video area within the container
    // (accounting for objectFit: 'contain' which maintains aspect ratio)
    const containerAspectRatio = rect.width / rect.height;
    const videoAspectRatio = videoWidth / videoHeight;
    
    let displayedVideoWidth, displayedVideoHeight, offsetX, offsetY;
    
    if (containerAspectRatio > videoAspectRatio) {
      // Container is wider than video - video fits height, centered horizontally
      displayedVideoHeight = rect.height;
      displayedVideoWidth = rect.height * videoAspectRatio;
      offsetX = (rect.width - displayedVideoWidth) / 2;
      offsetY = 0;
    } else {
      // Container is taller than video - video fits width, centered vertically
      displayedVideoWidth = rect.width;
      displayedVideoHeight = rect.width / videoAspectRatio;
      offsetX = 0;
      offsetY = (rect.height - displayedVideoHeight) / 2;
    }
    
    // Check if click is within the actual video area
    if (x < offsetX || x > offsetX + displayedVideoWidth || 
        y < offsetY || y > offsetY + displayedVideoHeight) {
      // Click is outside the video area (in the letterbox/pillarbox)
      return;
    }
    
    // Convert click coordinates to video coordinates
    const relativeX = (x - offsetX) / displayedVideoWidth;
    const relativeY = (y - offsetY) / displayedVideoHeight;
    
    const scaledX = Math.round(relativeX * videoWidth);
    const scaledY = Math.round(relativeY * videoHeight);
    
    // Ensure coordinates are within bounds
    const clampedX = Math.max(0, Math.min(scaledX, videoWidth - 1));
    const clampedY = Math.max(0, Math.min(scaledY, videoHeight - 1));
    
    onVideoClick({
      x: clampedX,
      y: clampedY,
      currentTime: currentTime,
      frameWidth: videoWidth,
      frameHeight: videoHeight
    });
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Video Element */}
      <Box 
        ref={videoContainerRef}
        sx={{ 
          position: 'relative', 
          flex: 1, 
          mb: 1.5,
          cursor: 'crosshair', // Show that video is clickable
          minHeight: 0 // Allow flex child to shrink
        }}
        onClick={handleVideoClick}
      >
        <video
          ref={videoRef}
          src={videoUrl}
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'contain',
            borderRadius: '8px',
            pointerEvents: 'none' // Let the container handle clicks
          }}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
        />
        
        {/* Play/Pause Overlay */}
        <Box
          sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            opacity: isPlaying ? 0 : 0.8,
            transition: 'opacity 0.3s',
            cursor: 'pointer',
            pointerEvents: 'auto' // Allow clicking on overlay
          }}
          onClick={(e) => {
            e.stopPropagation(); // Prevent triggering video click
            togglePlay();
          }}
        >
          <IconButton
            sx={{
              bgcolor: 'rgba(0, 0, 0, 0.7)',
              color: 'white',
              '&:hover': { bgcolor: 'rgba(0, 0, 0, 0.9)' }
            }}
            size="large"
          >
            {isPlaying ? <Pause /> : <PlayArrow />}
          </IconButton>
        </Box>

      </Box>

      {/* Controls */}
      <Paper sx={{ 
        p: { xs: 1.5, sm: 2 }, 
        bgcolor: 'background.paper',
        borderRadius: 2,
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
        flexShrink: 0 // Prevent controls from shrinking
      }}>
        {/* Click Instructions */}
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            mb: { xs: 1.5, sm: 2 },
            bgcolor: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            px: 3,
            py: 1.5,
            borderRadius: 2,
            fontSize: { xs: '0.9rem', sm: '1rem' },
            fontWeight: 600,
            pointerEvents: 'none',
            backdropFilter: 'blur(10px)',
            textAlign: 'center'
          }}
        >
          Click anywhere in the video to add a purchasable item in the shopping cart
        </Box>
        {/* Progress Bar */}
        <Box sx={{ mb: { xs: 1.5, sm: 2 } }}>
          <Slider
            value={currentTime}
            max={duration}
            onChange={handleSeek}
            sx={{
              '& .MuiSlider-thumb': {
                width: { xs: 14, sm: 16 },
                height: { xs: 14, sm: 16 },
                bgcolor: 'primary.main',
                '&:hover': {
                  boxShadow: '0 0 0 6px rgba(0, 0, 0, 0.1)',
                },
              },
              '& .MuiSlider-track': {
                height: { xs: 3, sm: 4 },
                bgcolor: 'primary.main',
                borderRadius: 2,
              },
              '& .MuiSlider-rail': {
                height: { xs: 3, sm: 4 },
                bgcolor: 'grey.200',
                borderRadius: 2,
              }
            }}
          />
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: { xs: 0.5, sm: 1 } }}>
            <Typography 
              variant="caption" 
              color="text.secondary"
              sx={{ fontWeight: 500, fontSize: { xs: '0.7rem', sm: '0.75rem' } }}
            >
              {formatTime(currentTime)}
            </Typography>
            <Typography 
              variant="caption" 
              color="text.secondary"
              sx={{ fontWeight: 500, fontSize: { xs: '0.7rem', sm: '0.75rem' } }}
            >
              {formatTime(duration)}
            </Typography>
          </Box>
        </Box>

        {/* Control Buttons */}
        <Stack direction="row" spacing={{ xs: 1, sm: 1.5 }} alignItems="center">
          <IconButton 
            onClick={togglePlay} 
            size="small"
            sx={{
              bgcolor: 'primary.main',
              color: 'white',
              width: { xs: 32, sm: 36 },
              height: { xs: 32, sm: 36 },
              '&:hover': {
                bgcolor: 'grey.800',
                transform: 'scale(1.05)'
              }
            }}
          >
            {isPlaying ? <Pause fontSize="small" /> : <PlayArrow fontSize="small" />}
          </IconButton>
          
          <Box sx={{ display: 'flex', alignItems: 'center', minWidth: { xs: 100, sm: 120 } }}>
            <IconButton 
              onClick={toggleMute} 
              size="small"
              sx={{
                color: 'text.secondary',
                width: { xs: 28, sm: 32 },
                height: { xs: 28, sm: 32 },
                '&:hover': {
                  bgcolor: 'grey.100'
                }
              }}
            >
              {isMuted ? <VolumeOff fontSize="small" /> : <VolumeUp fontSize="small" />}
            </IconButton>
            <Slider
              value={isMuted ? 0 : volume}
              onChange={handleVolumeChange}
              min={0}
              max={1}
              step={0.1}
              sx={{ 
                ml: { xs: 0.5, sm: 1 }, 
                width: { xs: 70, sm: 80 },
                '& .MuiSlider-thumb': {
                  width: { xs: 10, sm: 12 },
                  height: { xs: 10, sm: 12 },
                  bgcolor: 'primary.main',
                },
                '& .MuiSlider-track': {
                  height: { xs: 2, sm: 3 },
                  bgcolor: 'primary.main',
                },
                '& .MuiSlider-rail': {
                  height: { xs: 2, sm: 3 },
                  bgcolor: 'grey.200',
                }
              }}
            />
          </Box>

          <Box sx={{ ml: 'auto' }}>
            <IconButton 
              size="small"
              sx={{
                color: 'text.secondary',
                width: { xs: 28, sm: 32 },
                height: { xs: 28, sm: 32 },
                '&:hover': {
                  bgcolor: 'grey.100'
                }
              }}
            >
              <Fullscreen fontSize="small" />
            </IconButton>
          </Box>
        </Stack>
      </Paper>
    </Box>
  );
};

export default VideoPlayer;
