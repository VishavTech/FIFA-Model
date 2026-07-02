import { spawn, execSync } from 'child_process';
import fs from 'fs';

console.log('============================================================');
console.log('⚽ FIFA 2026 WINNER PREDICTOR: BOOTSTRAPPING PYTHON ENVIRONMENT ⚽');
console.log('============================================================\n');

// 1. Verify Python 3 Availability
let pythonCmd = 'python3';
try {
  const pythonVersion = execSync('python3 --version').toString().trim();
  console.log(`✅ Python 3 is available: ${pythonVersion}`);
} catch (err) {
  try {
    const pythonVersion = execSync('python --version').toString().trim();
    console.log(`✅ Python (as default) is available: ${pythonVersion}`);
    pythonCmd = 'python';
  } catch (err2) {
    console.error('❌ Critical Error: Python 3 was not detected in this environment.');
    process.exit(1);
  }
}

// 2. Bootstrap pip if not installed
let pipAvailable = false;
try {
  execSync(`${pythonCmd} -m pip --version`);
  pipAvailable = true;
  console.log('✅ pip is already installed.');
} catch (e) {
  console.log('⚠️ pip is not detected. Bootstrapping pip now...');
}

if (!pipAvailable) {
  // Method A: Try built-in ensurepip
  try {
    console.log('Executing python3 -m ensurepip...');
    execSync(`${pythonCmd} -m ensurepip --default-pip`, { stdio: 'inherit' });
    pipAvailable = true;
    console.log('✅ pip bootstrapped successfully via ensurepip!');
  } catch (ensureErr) {
    console.log('⚠️ ensurepip failed. Downloading get-pip.py as fallback...');
    // Method B: Download get-pip.py
    try {
      execSync('curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py', { stdio: 'inherit' });
      execSync(`${pythonCmd} get-pip.py --user`, { stdio: 'inherit' });
      pipAvailable = true;
      console.log('✅ pip bootstrapped successfully via get-pip.py!');
    } catch (getPipErr) {
      console.error('❌ Critical Error: Unable to bootstrap pip via any fallback.', getPipErr);
      process.exit(1);
    }
  }
}

// 3. Install requirements.txt
console.log('\n📦 Installing Python libraries from requirements.txt...');
try {
  execSync(`${pythonCmd} -m pip install -r requirements.txt`, { stdio: 'inherit' });
  console.log('✅ Python packages installed successfully.');
} catch (err) {
  console.log('⚠️ Standard pip install failed. Attempting with --user flag...');
  try {
    execSync(`${pythonCmd} -m pip install -r requirements.txt --user`, { stdio: 'inherit' });
    console.log('✅ Python packages installed successfully in user space.');
  } catch (userErr) {
    console.error('❌ Failed to install Python dependencies:', userErr);
    process.exit(1);
  }
}

// 4. Pre-execute Pipeline (Generates data, fits scaler, trains model, exports plots & results)
console.log('\n⚙️ Executing full Machine Learning Pipeline (main.py)...');
try {
  execSync(`${pythonCmd} main.py`, { stdio: 'inherit' });
  console.log('✅ Machine Learning Pipeline completed successfully.');
} catch (err) {
  console.error('❌ Error executing main.py:', err);
  // We can still try to boot the web app in case cache is present
}

// 5. Start Flask Web Server
console.log('\n🌐 Launching Flask Web Server on Port 3000...');
const appProcess = spawn(pythonCmd, ['app.py'], { stdio: 'inherit' });

appProcess.on('close', (code) => {
  console.log(`\n🛑 Flask server stopped. Exit code: ${code}`);
});

// Capture Termination Signals
process.on('SIGTERM', () => {
  console.log('SIGTERM signal received. Stopping Flask process...');
  appProcess.kill();
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('SIGINT signal received. Stopping Flask process...');
  appProcess.kill();
  process.exit(0);
});
