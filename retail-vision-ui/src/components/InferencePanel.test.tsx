import { render, act } from '@testing-library/react';
import { BrandProvider } from '../config/BrandContext';
import InferencePanel from './InferencePanel';

const originalFetch = global.fetch;
afterEach(() => { global.fetch = originalFetch; });

test('inference request includes the active brand key', async () => {
  const fetchMock = jest.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      timestamp: 1, video_time: 0, clicked_pixel: { x: 1, y: 1 },
      detections: [], frame_base64: '', annotated_frame_base64: '',
      clicked_object: null, inference_type: 'YOLO-E',
    }),
  });
  // @ts-ignore
  global.fetch = fetchMock;

  const click = { x: 1, y: 1, currentTime: 0, frameWidth: 10, frameHeight: 10 };

  await act(async () => {
    render(
      <BrandProvider>
        <InferencePanel lastClickData={click} onInference={() => {}} />
      </BrandProvider>
    );
  });

  expect(fetchMock).toHaveBeenCalled();
  const body = JSON.parse(fetchMock.mock.calls[0][1].body);
  expect(body.brand).toBe('blend360');
  expect(body.x).toBe(1);
});
