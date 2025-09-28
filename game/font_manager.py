"""
字體管理器 - 處理中文字體顯示，支持 Pygame 和 OpenGL 渲染
"""

import pygame
import os
import sys
from typing import Dict

try:
    from OpenGL.GL import (
        glEnable, glDisable, glBindTexture, glTexImage2D, glTexParameteri,
        glBegin, glEnd, glTexCoord2f, glVertex2f, glGenTextures, glDeleteTextures,
        GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE, GL_LINEAR, GL_TEXTURE_MAG_FILTER, 
        GL_TEXTURE_MIN_FILTER, GL_QUADS, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
        glBlendFunc, glPushMatrix, glPopMatrix, glLoadIdentity, glTranslatef
    )
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
from config import *

class FontManager:
    """字體管理器，支持中文字體和 OpenGL 渲染"""
    
    def __init__(self):
        self.fonts = {}
        self.default_font = None
        self.chinese_font = None
        self.texture_cache = {}  # Cache for OpenGL textures
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
    
    def get_font(self, size=32, bold=False) -> pygame.font.Font:
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
        """渲染文字 (Pygame Surface)"""
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
    
    def render_text_opengl(self, text, x, y, size=32, color=(1.0, 1.0, 1.0), bold=False):
        """渲染文字到 OpenGL (使用紋理)"""
        if not OPENGL_AVAILABLE:
            print(f"OpenGL not available, cannot render: {text}")
            return
            
        try:
            # Create cache key
            cache_key = (text, size, color, bold)
            
            # Check cache first
            if cache_key in self.texture_cache:
                texture_id, width, height = self.texture_cache[cache_key]
            else:
                # Render text to pygame surface
                surface = self.render_text(text, size, 
                                         (int(color[0]*255), int(color[1]*255), int(color[2]*255)), 
                                         bold)
                width, height = surface.get_size()
                
                # Convert to RGBA format for OpenGL
                texture_surface = pygame.Surface((width, height), pygame.SRCALPHA)
                texture_surface.blit(surface, (0, 0))
                texture_data = pygame.image.tostring(texture_surface, 'RGBA', True)
                
                # Generate OpenGL texture
                texture_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, texture_id)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
                
                # Cache the texture
                self.texture_cache[cache_key] = (texture_id, width, height)
            
            # Render the texture
            self._render_texture_quad(texture_id, x, y, width, height)
            
        except Exception as e:
            print(f"OpenGL text rendering failed: {e}")
    
    def _render_texture_quad(self, texture_id, x, y, width, height):
        """渲染紋理四邊形"""
        if not OPENGL_AVAILABLE:
            return
            
        try:
            # Enable blending for transparency
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            # Enable texturing
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            # Draw textured quad with flipped texture coordinates to fix upside-down text
            glBegin(GL_QUADS)
            glTexCoord2f(0, 1); glVertex2f(x, y)                    # Top-left: tex(0,1) -> screen(x,y)
            glTexCoord2f(1, 1); glVertex2f(x + width, y)            # Top-right: tex(1,1) -> screen(x+w,y)
            glTexCoord2f(1, 0); glVertex2f(x + width, y + height)   # Bottom-right: tex(1,0) -> screen(x+w,y+h)
            glTexCoord2f(0, 0); glVertex2f(x, y + height)           # Bottom-left: tex(0,0) -> screen(x,y+h)
            glEnd()
            
            # Cleanup
            glDisable(GL_TEXTURE_2D)
            glDisable(GL_BLEND)
            
        except Exception as e:
            print(f"Texture quad rendering failed: {e}")
    
    def render_text_list_opengl(self, text_list, start_x, start_y, line_height=30, size=24, color=(1.0, 1.0, 1.0)):
        """渲染文字列表到 OpenGL"""
        if not OPENGL_AVAILABLE:
            for i, text in enumerate(text_list):
                print(f"[{i}] {text}")
            return
            
        y = start_y
        for text in text_list:
            self.render_text_opengl(text, start_x, y, size, color)
            y += line_height
    
    def cleanup_textures(self):
        """清理 OpenGL 紋理快取"""
        if not OPENGL_AVAILABLE:
            return
            
        try:
            texture_ids = [texture_id for texture_id, _, _ in self.texture_cache.values()]
            if texture_ids:
                glDeleteTextures(texture_ids)
            self.texture_cache.clear()
        except Exception as e:
            print(f"Texture cleanup failed: {e}")

# 全局字體管理器實例
font_manager = None

def get_font_manager():
    """獲取全局字體管理器"""
    global font_manager
    if font_manager is None:
        font_manager = FontManager()
    return font_manager