var express = require('express');
var app = express();
const path = require('path');
const ejs = require('ejs');
const multer = require('multer');
var fs = require('fs');
const {spawn } = require('child_process');

obj= { Channel: 'OyeKool' }
const childPython= spawn('python', ['codespace.py', JSON.stringify(obj)]);
//const childPython= spawn('python', ['codespace.py']);
//const childPython= spawn('python', ['codespace.py', 'Visit', 'OyeKool')]);



childPython.stdout.on('data', (data) => {
  console.log(`stdout: ${data}`);
});

childPython.stderr.on('data', (data) => {
  console.error(`stderr: ${data}`);
});

childPython.on('close', (code) => {
  console.log(`child process exited with code: ${code}`);
});

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads')
  },
  filename: (req, file, cb) => {
    cb(null, "imagu" + path.extname(file.originalname))
    //Date.now() + path.extname(file.originalname)
  }
})

app.use(express.static('./images'));
app.use(express.static('./js'));

app.set("view engine", "ejs");

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '/index.html'));
})

const upload = multer ({storage: storage})

app.get('/upload', (req, res) => {
  res.render("upload")
});

app.post('/upload', upload.single("image"), (req, res) => {
  res.send("Image Uploaded")
});

app.listen(3000, function () {
  console.log('Example app listening on port 3000!');
});


