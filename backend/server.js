const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

const app = express();
app.use(cors());
app.use(express.json());

// 📂 Create uploads folder if not exists
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

// 📸 Multer setup for file upload
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, Date.now() + path.extname(file.originalname))
});
const upload = multer({ storage });

// 🟢 Route to upload video
app.post('/upload', upload.single('video'), (req, res) => {
  if (!req.file) return res.json({ success: false, error: 'No file uploaded' });
  return res.json({ success: true, filename: req.file.filename });
});

// 🧠 Fake AI processing route (can later add YOLO/OpenCV)
app.post('/process_video/:filename', (req, res) => {
  const { filename } = req.params;
  const filePath = path.join(uploadDir, filename);
  if (!fs.existsSync(filePath)) {
    return res.json({ success: false, error: 'File not found' });
  }

  // Simulate detection
  const peopleCount = Math.floor(Math.random() * 100);
  const processingTime = (Math.random() * 5).toFixed(2);
  const accuracy = (70 + Math.random() * 20).toFixed(2);

  res.json({
    success: true,
    people_count: peopleCount,
    processing_time: processingTime,
    accuracy: accuracy
  });
});

// 🔥 Start server
const PORT = 5000;
app.listen(PORT, () => console.log(`✅ Server running on http://localhost:${PORT}`));
