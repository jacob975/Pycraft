"""
字體管理器 - 處理中文字體顯示
"""

import pygame
import os
import sys

class FontManager:
    """字體管理器，支持中文字體"""
    
    def __init__(self):
        self.fonts = {}
        self.default_font = None
        self.chinese_font = None
        self._init_fonts()
    
    def _init_fonts(self):
        """初始化字體"""
        # 嘗試找到支持中文的系統字體
        chinese_font_names = [
            "SimHei",           # 黑體 (Windows)
            "Microsoft YaHei", # 微軟雅黑 (Windows)
            "SimSun",          # 宋體 (Windows)
            "PingFang SC",     # 蘋方 (macOS)
            "STHeiti",         # 華文黑體 (macOS)
            "Noto Sans CJK SC", # Noto Sans (Linux)
            "WenQuanYi Micro Hei", # 文泉驛微米黑 (Linux)
        ]
        
        # Windows系統字體路径
        windows_font_paths = [
            "C:/Windows/Fonts/simhei.ttf",      # 黑體
            "C:/Windows/Fonts/msyh.ttf",        # 微軟雅黑
            "C:/Windows/Fonts/simsun.ttc",      # 宋體
            "C:/Windows/Fonts/simkai.ttf",      # 楷體
        ]
        
        # 首先嘗試從文件路径加載字體
        for font_path in windows_font_paths:
            if os.path.exists(font_path):
                try:
                    self.chinese_font = font_path
                    print(f"找到中文字體: {font_path}")
                    break
                except:
                    continue
        
        # 如果找不到文件路径，嘗試系統字體名稱
        if not self.chinese_font:
            for font_name in chinese_font_names:
                if font_name in pygame.font.get_fonts():
                    self.chinese_font = font_name
                    print(f"找到系統中文字體: {font_name}")
                    break
        
        # 如果還是找不到，使用默認字體
        if not self.chinese_font:
            print("警告: 未找到中文字體，將使用默認字體 (中文可能顯示為方塊)")
            self.chinese_font = None
    
    def get_font(self, size=32, bold=False):
        """獲取字體對象"""
        font_key = (size, bold, self.chinese_font)
        
        if font_key not in self.fonts:
            try:
                if self.chinese_font and os.path.exists(str(self.chinese_font)):
                    # 從文件加載字體
                    self.fonts[font_key] = pygame.font.Font(self.chinese_font, size)
                elif self.chinese_font:
                    # 從系統字體加載
                    self.fonts[font_key] = pygame.font.SysFont(self.chinese_font, size, bold=bold)
                else:
                    # 使用默認字體
                    self.fonts[font_key] = pygame.font.Font(None, size)
            except:
                # 如果加載失敗，使用默認字體
                self.fonts[font_key] = pygame.font.Font(None, size)
        
        return self.fonts[font_key]
    
    def render_text(self, text, size=32, color=(255, 255, 255), bold=False):
        """渲染文字"""
        font = self.get_font(size, bold)
        try:
            return font.render(text, True, color)
        except:
            # 如果渲染失敗，嘗試用英文替代
            fallback_text = text.encode('ascii', 'ignore').decode('ascii')
            if fallback_text:
                return font.render(fallback_text, True, color)
            else:
                return font.render("Text Error", True, color)

# 全局字體管理器實例
font_manager = None

def get_font_manager():
    """獲取全局字體管理器"""
    global font_manager
    if font_manager is None:
        font_manager = FontManager()
    return font_manager