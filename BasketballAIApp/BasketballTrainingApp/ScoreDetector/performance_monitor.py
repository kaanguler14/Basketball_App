# performance_monitor.py
"""
Performans ölçümü ve izleme modülü
Real-time ve post-process performans analizi sağlar
"""

import time
import numpy as np
import cv2
from collections import defaultdict


class PerformanceMonitor:
    """
    Component-based performans ölçümü ve görselleştirme sınıfı
    
    Kullanım:
        perf = PerformanceMonitor()
        
        # Timing kaydet
        with perf.measure("component_name"):
            # kod
        
        # Manuel timing
        t0 = time.time()
        # kod
        perf.record("component_name", time.time() - t0)
        
        # Görselleştirme
        perf.draw_stats(frame)
        
        # Özet rapor
        perf.print_summary()
    """
    
    def __init__(self, enabled=True, window_size=30):
        """
        Args:
            enabled: Performans ölçümünü aktif et/kapat
            window_size: Kaç frame'lik ortalama alınacak (default: 30)
        """
        self.enabled = enabled
        self.window_size = window_size
        
        # Timing verileri
        self.timings = defaultdict(list)  # Component bazlı timing'ler
        self.frame_times = []  # Toplam frame süreleri
        
        # Current frame tracking
        self.frame_start_time = None
    
    def start_frame(self):
        """Yeni frame başlangıcını işaretle"""
        self.frame_start_time = time.time()
    
    def end_frame(self):
        """Frame sonunu işaretle ve toplam süreyi kaydet"""
        if self.frame_start_time and self.enabled:
            frame_time = time.time() - self.frame_start_time
            self.frame_times.append(frame_time)
            
            # Window size'ı korumak için eski değerleri sil
            if len(self.frame_times) > self.window_size:
                self.frame_times.pop(0)
    
    def record(self, component_name, duration):
        """
        Bir component'in timing'ini kaydet
        
        Args:
            component_name: Component adı (örn: "ball_detection")
            duration: Süre (saniye)
        """
        if not self.enabled:
            return
        
        self.timings[component_name].append(duration)
        
        # Window size'ı korumak için eski değerleri sil
        if len(self.timings[component_name]) > self.window_size:
            self.timings[component_name].pop(0)
    
    def measure(self, component_name):
        """
        Context manager olarak kullanım için
        
        Kullanım:
            with perf.measure("my_component"):
                # kod
        """
        return _TimingContext(self, component_name)
    
    def get_average(self, component_name):
        """Bir component'in ortalama süresini döndür (ms)"""
        if component_name not in self.timings or not self.timings[component_name]:
            return 0.0
        return np.mean(self.timings[component_name]) * 1000
    
    def get_min(self, component_name):
        """Bir component'in minimum süresini döndür (ms)"""
        if component_name not in self.timings or not self.timings[component_name]:
            return 0.0
        return np.min(self.timings[component_name]) * 1000
    
    def get_max(self, component_name):
        """Bir component'in maksimum süresini döndür (ms)"""
        if component_name not in self.timings or not self.timings[component_name]:
            return 0.0
        return np.max(self.timings[component_name]) * 1000
    
    def get_average_frame_time(self):
        """Ortalama frame süresini döndür (ms)"""
        if not self.frame_times:
            return 0.0
        return np.mean(self.frame_times) * 1000
    
    def get_average_fps(self):
        """Ortalama FPS'i döndür"""
        avg_frame = self.get_average_frame_time()
        if avg_frame > 0:
            return 1000.0 / avg_frame
        return 0.0
    
    def toggle(self):
        """Performans ölçümünü aç/kapat"""
        self.enabled = not self.enabled
        status = "AÇIK" if self.enabled else "KAPALI"
        print(f"📊 Performans İzleme: {status}")
        return self.enabled
    
    def draw_stats(self, frame, components=None):
        """
        Performans istatistiklerini frame üzerine çiz
        
        Args:
            frame: OpenCV frame (numpy array)
            components: Gösterilecek component'ler [(label, key, color), ...]
                       None ise default component'ler kullanılır
        """
        if not self.enabled or not self.timings:
            return
        
        # Default components
        if components is None:
            components = [
                ("Ball/Hoop Detection", "ball_hoop_detection", (255, 200, 100)),
                ("Player Tracking", "player_tracking", (100, 200, 255)),
                ("Shot Detection", "shot_detection", (255, 100, 200)),
                ("Minimap", "minimap", (200, 100, 255)),
                ("Rendering", "rendering", (100, 255, 200)),
                ("Scoreboard", "scoreboard", (255, 255, 100)),
            ]
        
        # Panel boyutu ve pozisyonu
        panel_width = 350
        panel_height = min(250, 80 + len(components) * 25)
        x_start = frame.shape[1] - panel_width - 20
        y_start = 20
        
        # Yarı saydam arka plan
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), 
                     (x_start + panel_width, y_start + panel_height), 
                     (20, 20, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Çerçeve
        cv2.rectangle(frame, (x_start, y_start), 
                     (x_start + panel_width, y_start + panel_height), 
                     (100, 200, 255), 2)
        
        # Başlık
        cv2.putText(frame, "PERFORMANCE STATS (P)", 
                   (x_start + 10, y_start + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Toplam frame süresi
        if self.frame_times:
            avg_frame = self.get_average_frame_time()
            avg_fps = self.get_average_fps()
            cv2.putText(frame, f"Total: {avg_frame:.1f}ms ({avg_fps:.1f} FPS)", 
                       (x_start + 10, y_start + 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        # Component bazlı timing'ler
        y_offset = 80
        for label, key, color in components:
            if key in self.timings and self.timings[key]:
                avg_time = self.get_average(key)
                
                # Label ve değerler
                cv2.putText(frame, f"{label}:", 
                           (x_start + 10, y_start + y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                
                cv2.putText(frame, f"{avg_time:.1f}ms", 
                           (x_start + 220, y_start + y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                
                # Progress bar (max 100ms için normalize)
                bar_width = int((avg_time / 100.0) * 100)
                bar_width = min(bar_width, 100)
                cv2.rectangle(frame, 
                             (x_start + 220, y_start + y_offset + 5), 
                             (x_start + 220 + bar_width, y_start + y_offset + 10), 
                             color, -1)
                
                y_offset += 25
    
    def print_summary(self):
        """Program sonunda detaylı performans özetini yazdır"""
        print("\n" + "="*60)
        print("📊 PERFORMANS ÖZETİ")
        print("="*60)
        
        # Toplam frame istatistikleri
        if self.frame_times:
            avg_frame = self.get_average_frame_time()
            min_frame = np.min(self.frame_times) * 1000
            max_frame = np.max(self.frame_times) * 1000
            avg_fps = self.get_average_fps()
            
            print(f"\n⏱️  Total Frame Time:")
            print(f"   Ortalama: {avg_frame:.2f}ms")
            print(f"   Min: {min_frame:.2f}ms")
            print(f"   Max: {max_frame:.2f}ms")
            print(f"   Ortalama FPS: {avg_fps:.1f}")
        
        # Component bazlı istatistikler
        if self.timings:
            print(f"\n🔧 Component Timing'leri:")
            
            # Sorting: En yavaştan en hızlıya
            sorted_components = sorted(
                self.timings.items(), 
                key=lambda x: np.mean(x[1]) if x[1] else 0, 
                reverse=True
            )
            
            total_frame_time = np.mean(self.frame_times) if self.frame_times else 0
            
            for component, timings in sorted_components:
                if timings:
                    avg_time = np.mean(timings) * 1000
                    min_time = np.min(timings) * 1000
                    max_time = np.max(timings) * 1000
                    
                    # Yüzdelik hesaplama
                    percentage = (avg_time / (total_frame_time * 1000)) * 100 if total_frame_time > 0 else 0
                    
                    print(f"\n   {component}:")
                    print(f"      Ortalama: {avg_time:.2f}ms ({percentage:.1f}%)")
                    print(f"      Min: {min_time:.2f}ms")
                    print(f"      Max: {max_time:.2f}ms")
        
        print("\n" + "="*60 + "\n")
    
    def export_to_csv(self, filename="performance_log.csv"):
        """Performans verilerini CSV dosyasına kaydet"""
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Component', 'Avg(ms)', 'Min(ms)', 'Max(ms)', 'Samples'])
            
            # Frame timing
            if self.frame_times:
                writer.writerow([
                    'Total Frame',
                    f"{self.get_average_frame_time():.2f}",
                    f"{np.min(self.frame_times) * 1000:.2f}",
                    f"{np.max(self.frame_times) * 1000:.2f}",
                    len(self.frame_times)
                ])
            
            # Component timings
            for component, timings in sorted(self.timings.items()):
                if timings:
                    writer.writerow([
                        component,
                        f"{self.get_average(component):.2f}",
                        f"{self.get_min(component):.2f}",
                        f"{self.get_max(component):.2f}",
                        len(timings)
                    ])
        
        print(f"📄 Performans verileri '{filename}' dosyasına kaydedildi.")
    
    def reset(self):
        """Tüm timing verilerini sıfırla"""
        self.timings.clear()
        self.frame_times.clear()
        print("🔄 Performans verileri sıfırlandı.")


class _TimingContext:
    """Context manager for timing measurements"""
    
    def __init__(self, monitor, component_name):
        self.monitor = monitor
        self.component_name = component_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitor.record(self.component_name, duration)
        return False


# Convenience functions
def create_monitor(enabled=True, window_size=30):
    """Factory function to create a PerformanceMonitor"""
    return PerformanceMonitor(enabled=enabled, window_size=window_size)


