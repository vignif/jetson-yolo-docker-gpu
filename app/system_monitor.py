"""System monitoring for Jetson device."""
import os
import subprocess
from typing import Dict, Optional


class SystemMonitor:
    """Monitor Jetson system stats like temperature, fan, power mode."""
    
    def __init__(self):
        """Initialize system monitor."""
        self.tegrastats_available = self._check_tegrastats()
    
    def _check_tegrastats(self) -> bool:
        """Check if tegrastats is available."""
        try:
            result = subprocess.run(
                ['which', 'tegrastats'],
                capture_output=True,
                timeout=1
            )
            return result.returncode == 0
        except:
            return False
    
    def get_temperature(self) -> Dict[str, Optional[float]]:
        """Get device temperatures in Celsius.
        
        Returns:
            Dictionary with temperature readings
        """
        temps = {}
        
        # Try thermal zones (standard Linux)
        thermal_zones = [
            ('/sys/class/thermal/thermal_zone0/temp', 'CPU'),
            ('/sys/class/thermal/thermal_zone1/temp', 'GPU'),
            ('/sys/class/thermal/thermal_zone2/temp', 'AUX'),
        ]
        
        for path, name in thermal_zones:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        # Temperature is in millidegrees
                        temp = int(f.read().strip()) / 1000.0
                        temps[name.lower()] = round(temp, 1)
            except:
                pass
        
        return temps
    
    def get_fan_speed(self) -> Optional[int]:
        """Get current fan speed (0-255).
        
        Returns:
            Fan speed or None if not available
        """
        fan_path = '/sys/devices/pwm-fan/target_pwm'
        
        try:
            if os.path.exists(fan_path):
                with open(fan_path, 'r') as f:
                    speed = int(f.read().strip())
                    return speed
        except:
            pass
        
        return None
    
    def get_power_mode(self) -> Optional[str]:
        """Get current power mode (MAXN, 5W, etc.).
        
        Returns:
            Power mode string or None
        """
        try:
            result = subprocess.run(
                ['nvpmodel', '-q'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                # Parse output like "NV Power Mode: MAXN"
                for line in result.stdout.split('\n'):
                    if 'NV Power Mode:' in line:
                        mode = line.split(':')[1].strip()
                        return mode
        except:
            pass
        
        return None
    
    def get_cpu_usage(self) -> Optional[float]:
        """Get overall CPU usage percentage.
        
        Returns:
            CPU usage percentage or None
        """
        try:
            # Read /proc/stat for CPU usage
            with open('/proc/stat', 'r') as f:
                line = f.readline()
                # Format: cpu  user nice system idle iowait irq softirq
                fields = line.split()
                if fields[0] == 'cpu':
                    idle = int(fields[4])
                    total = sum(int(x) for x in fields[1:])
                    
                    # Calculate percentage (need previous values for accurate reading)
                    # For now, just return a basic stat
                    if not hasattr(self, '_prev_idle'):
                        self._prev_idle = idle
                        self._prev_total = total
                        return None
                    
                    idle_delta = idle - self._prev_idle
                    total_delta = total - self._prev_total
                    
                    self._prev_idle = idle
                    self._prev_total = total
                    
                    if total_delta > 0:
                        usage = 100.0 * (1.0 - idle_delta / total_delta)
                        return round(usage, 1)
        except:
            pass
        
        return None
    
    def get_uptime(self) -> Optional[str]:
        """Get system uptime.
        
        Returns:
            Uptime string or None
        """
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.read().split()[0])
                
                days = int(uptime_seconds // 86400)
                hours = int((uptime_seconds % 86400) // 3600)
                minutes = int((uptime_seconds % 3600) // 60)
                
                if days > 0:
                    return f"{days}d {hours}h {minutes}m"
                elif hours > 0:
                    return f"{hours}h {minutes}m"
                else:
                    return f"{minutes}m"
        except:
            pass
        
        return None
    
    def get_all_stats(self) -> Dict:
        """Get all system statistics.
        
        Returns:
            Dictionary with all available stats
        """
        stats = {
            "temperature": self.get_temperature(),
            "fan_speed": self.get_fan_speed(),
            "power_mode": self.get_power_mode(),
            "cpu_usage": self.get_cpu_usage(),
            "uptime": self.get_uptime()
        }
        
        return stats
