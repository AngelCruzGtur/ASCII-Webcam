import cv2
import tkinter as tk
import numpy as np
import threading
import sys
import time

# ASCII characters for brightness mapping
ASCII_CHARS = "@%#*+=-:. "

# Resize the frame proportionally for ASCII conversion
def resizeFrame(frame, newWidth=150):
	height, width = frame.shape[:2]
	aspectRatio = height / width
	newHeight = int(newWidth * aspectRatio * 0.55)
	return cv2.resize(frame, (newWidth, newHeight), interpolation=cv2.INTER_NEAREST)

# Apply sharpening and edge emphasis
def enhanceFrame(frame):
	blurred = cv2.GaussianBlur(frame, (3, 3), 0)
	gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
	edges = cv2.Laplacian(gray, cv2.CV_64F)
	edges = cv2.convertScaleAbs(edges)
	_, edgeMask = cv2.threshold(edges, 20, 255, cv2.THRESH_BINARY)
	edgeMaskColor = cv2.merge([edgeMask] * 3)

	sharpenKernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
	sharpened = cv2.filter2D(blurred, -1, sharpenKernel)
	return np.where(edgeMaskColor == 255, sharpened, blurred)

# Apply gamma correction for brightness control
def applyGamma(frame, gamma=1.3):
	invGamma = 1.0 / gamma
	table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
	return cv2.LUT(frame, table)

# Reduce color precision for ASCII styling
def compressColor(r, g, b):
	r = (r // 64) * 64
	g = (g // 64) * 64
	b = (b // 64) * 64
	return f'#{r:02x}{g:02x}{b:02x}'

# Threaded class to continuously fetch frames from webcam
class FrameFetcher(threading.Thread):
	def __init__(self, camIndex=0):
		super().__init__()
		self.cam = cv2.VideoCapture(camIndex)
		self.lock = threading.Lock()
		self.currentFrame = None
		self.isRunning = True

	def run(self):
		while self.isRunning:
			success, frame = self.cam.read()
			if success:
				with self.lock:
					self.currentFrame = frame
			else:
				break
			time.sleep(0.005)

	def getFrame(self):
		with self.lock:
			return self.currentFrame.copy() if self.currentFrame is not None else None

	def stop(self):
		self.isRunning = False
		self.cam.release()

# Draw ASCII video to canvas and refresh in loop
def renderAscii(canvas, root, fetcher):
	lastFrame = {}
	charMap = {}
	cellWidth, cellHeight = 5, 9

	def updateCanvas():
		frame = fetcher.getFrame()
		if frame is None:
			root.after(1, updateCanvas)
			return

		frame = applyGamma(frame)
		resized = resizeFrame(frame, newWidth=150)
		processed = enhanceFrame(resized)
		gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
		rows, cols = gray.shape

		marginX = max(0, (canvas.winfo_width() - cols * cellWidth) // 2)
		marginY = max(0, (canvas.winfo_height() - rows * cellHeight) // 2)

		for y in range(rows):
			for x in range(cols):
				brightness = gray[y, x]
				if brightness < 10:
					continue

				b, g, r = processed[y, x]
				color = compressColor(r, g, b)
				idx = int((brightness - 10) / (255 - 10) * (len(ASCII_CHARS) - 1))
				idx = max(0, min(len(ASCII_CHARS) - 1, idx))
				char = ASCII_CHARS[idx]

				pos = (x, y)
				if lastFrame.get(pos) == (char, color):
					continue
				lastFrame[pos] = (char, color)

				if pos in charMap:
					canvas.itemconfig(charMap[pos], text=char, fill=color)
				else:
					charMap[pos] = canvas.create_text(
						marginX + x * cellWidth,
						marginY + y * cellHeight,
						text=char,
						fill=color,
						font="TkFixedFont",
						anchor="nw",
						tags="asciiFrame"
					)

		root.after(1, updateCanvas)

	updateCanvas()

# Launch the app
def main():
	root = tk.Tk()
	root.title("Live ASCII Camera Viewer")
	root.configure(bg="black")
	root.geometry("900x600")
	root.minsize(300, 200)

	canvas = tk.Canvas(root, bg="black", highlightthickness=0)
	canvas.pack(expand=True, fill=tk.BOTH)

	fetcher = FrameFetcher(camIndex=0)
	fetcher.start()

	def onClose():
		fetcher.stop()
		root.destroy()
		sys.exit()

	root.protocol("WM_DELETE_WINDOW", onClose)

	renderAscii(canvas, root, fetcher)
	root.mainloop()

if __name__ == '__main__':
	main()
