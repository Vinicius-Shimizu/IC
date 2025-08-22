import scallopy
import google.generativeai as genai
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for headless environments
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import re

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
program = '''
        type direction = UP | DOWN | LEFT | RIGHT

        type board(x: i32, y: i32)
        rel board = {
            (0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8),
            (0, 7), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7),
            (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (7, 6), (8, 6),
            (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5), (8, 5),
            (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4),
            (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 3),
            (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2),
            (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1),
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0)
        }

        rel empty(x, y) = board(x, y) and not enemies(x, y) and not pawn(x, y) and not gold(x, y) and not silver(x, y) and not lance(x, y) and not rook(x, y) and not promoted_rook(x, y) and not bishop(x, y) and not promoted_bishop(x, y) and not knight(x, y) and not king(x, y)

        rel pawn_moves(x, y, x, y + 1) = pawn(x, y) and y < 8 and (empty(x, y + 1) or enemies(x, y + 1))

        // Gold general
        rel gold_moves(x, y, x, y + 1) = gold(x, y) and y < 8 and (empty(x, y + 1) or enemies(x, y + 1))
        rel gold_moves(x, y, x - 1, y + 1) = gold(x, y) and y < 8 and x > 0 and (empty(x - 1, y + 1) or enemies(x - 1, y + 1)) 
        rel gold_moves(x, y, x + 1, y + 1) = gold(x, y) and y < 8 and x < 8 and (empty(x + 1, y + 1) or enemies(x + 1, y + 1))
        rel gold_moves(x, y, x - 1, y) = gold(x, y) and x > 0 and (empty(x - 1, y) or enemies(x - 1, y))
        rel gold_moves(x, y, x + 1, y) = gold(x, y) and x < 8 and (empty(x + 1, y) or enemies(x + 1, y))
        rel gold_moves(x, y, x, y - 1) = gold(x, y) and y > 0 and (empty(x, y - 1) or enemies(x, y - 1))

        // Silver general
        rel silver_moves(x, y, x, y + 1) = silver(x, y) and y < 8 and (empty(x, y + 1) or enemies(x, y + 1))
        rel silver_moves(x, y, x - 1, y + 1) = silver(x, y) and y < 8 and x > 0 and (empty(x - 1, y + 1) or enemies(x - 1, y + 1))
        rel silver_moves(x, y, x + 1, y + 1) = silver(x, y) and y < 8 and x < 8 and (empty(x + 1, y + 1) or enemies(x + 1, y + 1))
        rel silver_moves(x, y, x - 1, y - 1) = silver(x, y) and y > 0 and x > 0 and (empty(x - 1, y - 1) or enemies(x - 1, y - 1))
        rel silver_moves(x, y, x + 1, y - 1) = silver(x, y) and y > 0 and x < 8 and (empty(x + 1, y - 1) or enemies(x + 1, y - 1))

        // Lance
        rel lance_moves_empty(x, y, x, y + 1) = lance(x, y) and y < 8 and empty(x, y + 1)
        rel lance_moves_empty(x, y, x, y2) = lance_moves_empty(x, y, x, y1) and y1 < 8 and empty(x, y1 + 1) and y2 == y1 + 1
        rel lance_moves_enemy(x, y, x, y2) = lance_moves_empty(x, y, x, y1) and y1 < 8 and enemies(x, y1 + 1) and y2 == y1 + 1
        rel lance_moves_adjacent(x, y, x, y + 1) = lance(x, y) and enemies(x, y + 1) and y < 8

        rel lance_moves(xsrc, ysrc, xdst, ydst) = lance_moves_empty(xsrc, ysrc, xdst, ydst) or lance_moves_enemy(xsrc, ysrc, xdst, ydst) or lance_moves_adjacent(xsrc, ysrc, xdst, ydst)


        // Rook
        // UP
        rel rook_line_empty(x, y, x, y + 1, UP) = rook(x, y) and y < 8 and empty(x, y + 1)
        rel rook_line_empty(x, y, x, y2, UP) = rook_line_empty(x, y, x, y1, UP) and y1 < 8 and empty(x, y1 + 1) and y2 == y1 + 1
        rel rook_line_enemies(x, y, x, y2, UP) = rook_line_empty(x, y, x, y1, UP) and y1 < 8 and enemies(x, y1 + 1) and y2 == y1 + 1
        rel rook_line_adjacent(x, y, x, y + 1, UP) = rook(x, y) and enemies(x, y + 1) and y < 8
        // DOWN
        rel rook_line_empty(x, y, x, y - 1, DOWN) = rook(x, y) and y > 0 and empty(x, y - 1)
        rel rook_line_empty(x, y, x, y2, DOWN) = rook_line_empty(x, y, x, y1, DOWN) and y1 > 0 and empty(x, y1 - 1) and y2 == y1 - 1
        rel rook_line_enemies(x, y, x, y2, DOWN) = rook_line_empty(x, y, x, y1, DOWN) and y1 > 0 and enemies(x, y1 - 1) and y2 == y1 - 1
        rel rook_line_adjacent(x, y, x, y - 1, DOWN) = rook(x, y) and enemies(x, y - 1) and y > 0
        // RIGHT
        rel rook_line_empty(x, y, x + 1, y, RIGHT) = rook(x, y) and x < 8 and empty(x + 1, y)
        rel rook_line_empty(x, y, x2, y, RIGHT) = rook_line_empty(x, y, x1, y, RIGHT) and x1 < 8 and empty(x1 + 1, y) and x2 == x1 + 1
        rel rook_line_enemies(x, y, x2, y, RIGHT) = rook_line_empty(x, y, x1, y, RIGHT) and x1 < 8 and enemies(x1 + 1, y) and x2 == x1 + 1
        rel rook_line_adjacent(x, y, x + 1, y, RIGHT) = rook(x, y) and enemies(x + 1, y) and x < 8
        // LEFT
        rel rook_line_empty(x, y, x - 1, y, LEFT) = rook(x, y) and x > 0 and empty(x - 1, y)
        rel rook_line_empty(x, y, x2, y, LEFT) = rook_line_empty(x, y, x1, y, LEFT) and x1 > 0 and empty(x1 - 1, y) and x2 == x1 - 1
        rel rook_line_enemies(x, y, x2, y, LEFT) = rook_line_empty(x, y, x1, y, LEFT) and x1 > 0 and enemies(x1 - 1, y) and x2 == x1 - 1
        rel rook_line_adjacent(x, y, x - 1, y, LEFT) = rook(x, y) and enemies(x - 1, y) and x > 0

        rel rook_moves(x1, y1, x2, y2) = rook_line_empty(x1, y1, x2, y2, UP) or rook_line_enemies(x1, y1, x2, y2, UP) or rook_line_adjacent(x1, y1, x2, y2, UP)
        rel rook_moves(x1, y1, x2, y2) = rook_line_empty(x1, y1, x2, y2, DOWN) or rook_line_enemies(x1, y1, x2, y2, DOWN) or rook_line_adjacent(x1, y1, x2, y2, DOWN)
        rel rook_moves(x1, y1, x2, y2) = rook_line_empty(x1, y1, x2, y2, RIGHT) or rook_line_enemies(x1, y1, x2, y2, RIGHT) or rook_line_adjacent(x1, y1, x2, y2, RIGHT)
        rel rook_moves(x1, y1, x2, y2) = rook_line_empty(x1, y1, x2, y2, LEFT) or rook_line_enemies(x1, y1, x2, y2, LEFT) or rook_line_adjacent(x1, y1, x2, y2, LEFT)

        // Promoted Rook
        // UP
        rel promoted_rook_line_empty(x, y, x, y + 1, UP) = promoted_rook(x, y) and y < 8 and empty(x, y + 1)
        rel promoted_rook_line_empty(x, y, x, y2, UP) = promoted_rook_line_empty(x, y, x, y1, UP) and y1 < 8 and empty(x, y1 + 1) and y2 == y1 + 1
        rel promoted_rook_line_enemies(x, y, x, y2, UP) = promoted_rook_line_empty(x, y, x, y1, UP) and y1 < 8 and enemies(x, y1 + 1) and y2 == y1 + 1
        rel promoted_rook_line_adjacent(x, y, x, y + 1, UP) = promoted_rook(x, y) and enemies(x, y + 1) and y < 8
        // DOWN
        rel promoted_rook_line_empty(x, y, x, y - 1, DOWN) = promoted_rook(x, y) and y > 0 and empty(x, y - 1)
        rel promoted_rook_line_empty(x, y, x, y2, DOWN) = promoted_rook_line_empty(x, y, x, y1, DOWN) and y1 > 0 and empty(x, y1 - 1) and y2 == y1 - 1
        rel promoted_rook_line_enemies(x, y, x, y2, DOWN) = promoted_rook_line_empty(x, y, x, y1, DOWN) and y1 > 0 and enemies(x, y1 - 1) and y2 == y1 - 1
        rel promoted_rook_line_adjacent(x, y, x, y - 1, DOWN) = promoted_rook(x, y) and enemies(x, y - 1) and y > 0
        // RIGHT
        rel promoted_rook_line_empty(x, y, x + 1, y, RIGHT) = promoted_rook(x, y) and x < 8 and empty(x + 1, y)
        rel promoted_rook_line_empty(x, y, x2, y, RIGHT) = promoted_rook_line_empty(x, y, x1, y, RIGHT) and x1 < 8 and empty(x1 + 1, y) and x2 == x1 + 1
        rel promoted_rook_line_enemies(x, y, x2, y, RIGHT) = promoted_rook_line_empty(x, y, x1, y, RIGHT) and x1 < 8 and enemies(x1 + 1, y) and x2 == x1 + 1
        rel promoted_rook_line_adjacent(x, y, x + 1, y, RIGHT) = promoted_rook(x, y) and enemies(x + 1, y) and x < 8
        // LEFT
        rel promoted_rook_line_empty(x, y, x - 1, y, LEFT) = promoted_rook(x, y) and x > 0 and empty(x - 1, y)
        rel promoted_rook_line_empty(x, y, x2, y, LEFT) = promoted_rook_line_empty(x, y, x1, y, LEFT) and x1 > 0 and empty(x1 - 1, y) and x2 == x1 - 1
        rel promoted_rook_line_enemies(x, y, x2, y, LEFT) = promoted_rook_line_empty(x, y, x1, y, LEFT) and x1 > 0 and enemies(x1 - 1, y) and x2 == x1 - 1
        rel promoted_rook_line_adjacent(x, y, x - 1, y, LEFT) = promoted_rook(x, y) and enemies(x - 1, y) and x > 0

        rel promoted_rook_moves(x1, y1, x2, y2) = promoted_rook_line_empty(x1, y1, x2, y2, UP) or promoted_rook_line_enemies(x1, y1, x2, y2, UP) or promoted_rook_line_adjacent(x1, y1, x2, y2, UP)
        rel promoted_rook_moves(x1, y1, x2, y2) = promoted_rook_line_empty(x1, y1, x2, y2, DOWN) or promoted_rook_line_enemies(x1, y1, x2, y2, DOWN) or promoted_rook_line_adjacent(x1, y1, x2, y2, DOWN)
        rel promoted_rook_moves(x1, y1, x2, y2) = promoted_rook_line_empty(x1, y1, x2, y2, RIGHT) or promoted_rook_line_enemies(x1, y1, x2, y2, RIGHT) or promoted_rook_line_adjacent(x1, y1, x2, y2, RIGHT)
        rel promoted_rook_moves(x1, y1, x2, y2) = promoted_rook_line_empty(x1, y1, x2, y2, LEFT) or promoted_rook_line_enemies(x1, y1, x2, y2, LEFT) or promoted_rook_line_adjacent(x1, y1, x2, y2, LEFT)
        rel promoted_rook_moves(x, y, x + 1, y + 1) = promoted_rook(x, y) and (empty(x + 1, y + 1) or enemies(x + 1, y + 1))
        rel promoted_rook_moves(x, y, x - 1, y + 1) = promoted_rook(x, y) and (empty(x - 1, y + 1) or enemies(x - 1, y + 1))
        rel promoted_rook_moves(x, y, x + 1, y - 1) = promoted_rook(x, y) and (empty(x + 1, y - 1) or enemies(x + 1, y - 1))
        rel promoted_rook_moves(x, y, x - 1, y - 1) = promoted_rook(x, y) and (empty(x - 1, y - 1) or enemies(x - 1, y - 1))

        // Bishop
        // UP-RIGHT
        rel bishop_diag_empty(x, y, x + 1, y + 1, RIGHT, UP) = bishop(x, y) and x < 8 and y < 8 and empty(x + 1, y + 1)
        rel bishop_diag_empty(x, y, x2, y2, RIGHT, UP) = bishop_diag_empty(x, y, x1, y1, RIGHT, UP) and x1 < 8 and y1 < 8 and empty(x1 + 1, y1 + 1) and x2 == x1 + 1 and y2 == y1 + 1
        rel bishop_diag_enemies(x, y, x2, y2, RIGHT, UP) = bishop_diag_empty(x, y, x1, y1, RIGHT, UP) and x1 < 8 and y1 < 8 and enemies(x1 + 1, y1 + 1) and x2 == x1 + 1 and y2 == y1 + 1
        rel bishop_diag_adjacent(x, y, x + 1, y + 1, RIGHT, UP) = bishop(x, y) and enemies(x + 1, y + 1) and x < 8 and y < 8

        // // DOWN-RIGHT
        rel bishop_diag_empty(x, y, x + 1, y - 1, RIGHT, DOWN) = bishop(x, y) and x < 8 and y > 0 and empty(x + 1, y - 1)
        rel bishop_diag_empty(x, y, x2, y2, RIGHT, DOWN) = bishop_diag_empty(x, y, x1, y1, RIGHT, DOWN) and x1 < 8 and y1 > 0 and empty(x1 + 1, y1 - 1) and x2 == x1 + 1 and y2 == y1 - 1
        rel bishop_diag_enemies(x, y, x2, y2, RIGHT, DOWN) = bishop_diag_empty(x, y, x1, y1, RIGHT, DOWN) and x1 < 8 and y1 > 0 and enemies(x1 + 1, y1 - 1) and x2 == x1 + 1 and y2 == y1 - 1
        rel bishop_diag_adjacent(x, y, x + 1, y - 1, RIGHT, UP) = bishop(x, y) and enemies(x + 1, y - 1) and x < 8 and y > 0
        // // DOWN-LEFT
        rel bishop_diag_empty(x, y, x - 1, y - 1, LEFT, DOWN) = bishop(x, y) and x > 0 and y > 0 and empty(x - 1, y - 1)
        rel bishop_diag_empty(x, y, x2, y2, LEFT, DOWN) = bishop_diag_empty(x, y, x1, y1, LEFT, DOWN) and x1 > 0 and y1 > 0 and empty(x1 - 1, y1 - 1) and x2 == x1 - 1 and y2 == y1 - 1
        rel bishop_diag_enemies(x, y, x2, y2, RIGHT, DOWN) = bishop_diag_empty(x, y, x1, y1, RIGHT, DOWN) and x1 > 0 and y1 > 0 and enemies(x1 - 1, y1 - 1) and x2 == x1 - 1 and y2 == y1 - 1
        rel bishop_diag_adjacent(x, y, x - 1, y - 1, LEFT, UP) = bishop(x, y) and enemies(x - 1, y - 1) and x > 0 and y > 0
        // // UP-LEFT
        rel bishop_diag_empty(x, y, x - 1, y + 1, LEFT, UP) = bishop(x, y) and x > 0 and y < 8 and empty(x - 1, y + 1)
        rel bishop_diag_empty(x, y, x2, y2, LEFT, UP) = bishop_diag_empty(x, y, x1, y1, LEFT, UP) and x1 > 0 and y1 < 8 and empty(x1 - 1, y1 + 1) and x2 == x1 - 1 and y2 == y1 + 1
        rel bishop_diag_enemies(x, y, x2, y2, LEFT, UP) = bishop_diag_empty(x, y, x1, y1, LEFT, UP) and x1 > 0 and y1 < 8 and enemies(x1 - 1, y1 + 1) and x2 == x1 - 1 and y2 == y1 + 1
        rel bishop_diag_adjacent(x, y, x - 1, y + 1, LEFT, UP) = bishop(x, y) and enemies(x - 1, y + 1) and x > 0 and y < 8

        rel bishop_moves(x1, y1, x2, y2) = bishop_diag_empty(x1, y1, x2, y2, RIGHT, UP) or bishop_diag_enemies(x1, y1, x2, y2, RIGHT, UP) or bishop_diag_adjacent(x1, y1, x2, y2, RIGHT, UP)
        rel bishop_moves(x1, y1, x2, y2) = bishop_diag_empty(x1, y1, x2, y2, RIGHT, DOWN) or bishop_diag_enemies(x1, y1, x2, y2, RIGHT, DOWN) or bishop_diag_adjacent(x1, y1, x2, y2, RIGHT, DOWN)
        rel bishop_moves(x1, y1, x2, y2) = bishop_diag_empty(x1, y1, x2, y2, LEFT, UP) or bishop_diag_enemies(x1, y1, x2, y2, LEFT, UP) or bishop_diag_adjacent(x1, y1, x2, y2, LEFT, UP)
        rel bishop_moves(x1, y1, x2, y2) = bishop_diag_empty(x1, y1, x2, y2, LEFT, DOWN) or bishop_diag_enemies(x1, y1, x2, y2, LEFT, DOWN) or bishop_diag_adjacent(x1, y1, x2, y2, LEFT, DOWN)

        // Promoted Bishop
        // UP-RIGHT
        rel promoted_bishop_diag_empty(x, y, x + 1, y + 1, RIGHT, UP) = promoted_bishop(x, y) and x < 8 and y < 8 and empty(x + 1, y + 1)
        rel promoted_bishop_diag_empty(x, y, x + 1, y + 1, RIGHT, UP) = promoted_bishop(x, y) and x < 8 and y < 8 and empty(x + 1, y + 1)
        rel promoted_bishop_diag_empty(x, y, x2, y2, RIGHT, UP) = promoted_bishop_diag_empty(x, y, x1, y1, RIGHT, UP) and x1 < 8 and y1 < 8 and empty(x1 + 1, y1 + 1) and x2 == x1 + 1 and y2 == y1 + 1
        rel promoted_bishop_diag_enemies(x, y, x2, y2, RIGHT, UP) = promoted_bishop_diag_empty(x, y, x1, y1, RIGHT, UP) and x1 < 8 and y1 < 8 and enemies(x1 + 1, y1 + 1) and x2 == x1 + 1 and y2 == y1 + 1
        rel promoted_bishop_diag_adjacent(x, y, x + 1, y + 1, RIGHT, UP) = promoted_bishop(x, y) and enemies(x + 1, y + 1) and x < 8 and y < 8

        // // DOWN-RIGHT
        rel promoted_bishop_diag_empty(x, y, x + 1, y - 1, RIGHT, DOWN) = promoted_bishop(x, y) and x < 8 and y > 0 and empty(x + 1, y - 1)
        rel promoted_bishop_diag_empty(x, y, x2, y2, RIGHT, DOWN) = promoted_bishop_diag_empty(x, y, x1, y1, RIGHT, DOWN) and x1 < 8 and y1 > 0 and empty(x1 + 1, y1 - 1) and x2 == x1 + 1 and y2 == y1 - 1
        rel promoted_bishop_diag_enemies(x, y, x2, y2, RIGHT, DOWN) = promoted_bishop_diag_empty(x, y, x1, y1, RIGHT, DOWN) and x1 < 8 and y1 > 0 and enemies(x1 + 1, y1 - 1) and x2 == x1 + 1 and y2 == y1 - 1
        rel promoted_bishop_diag_adjacent(x, y, x + 1, y - 1, RIGHT, UP) = promoted_bishop(x, y) and enemies(x + 1, y - 1) and x < 8 and y > 0
        // // DOWN-LEFT
        rel promoted_bishop_diag_empty(x, y, x - 1, y - 1, LEFT, DOWN) = promoted_bishop(x, y) and x > 0 and y > 0 and empty(x - 1, y - 1)
        rel promoted_bishop_diag_empty(x, y, x2, y2, LEFT, DOWN) = promoted_bishop_diag_empty(x, y, x1, y1, LEFT, DOWN) and x1 > 0 and y1 > 0 and empty(x1 - 1, y1 - 1) and x2 == x1 - 1 and y2 == y1 - 1
        rel promoted_bishop_diag_enemies(x, y, x2, y2, RIGHT, DOWN) = promoted_bishop_diag_empty(x, y, x1, y1, RIGHT, DOWN) and x1 > 0 and y1 > 0 and enemies(x1 - 1, y1 - 1) and x2 == x1 - 1 and y2 == y1 - 1
        rel promoted_bishop_diag_adjacent(x, y, x - 1, y - 1, LEFT, UP) = promoted_bishop(x, y) and enemies(x - 1, y - 1) and x > 0 and y > 0
        // // UP-LEFT
        rel promoted_bishop_diag_empty(x, y, x - 1, y + 1, LEFT, UP) = promoted_bishop(x, y) and x > 0 and y < 8 and empty(x - 1, y + 1)
        rel promoted_bishop_diag_empty(x, y, x2, y2, LEFT, UP) = promoted_bishop_diag_empty(x, y, x1, y1, LEFT, UP) and x1 > 0 and y1 < 8 and empty(x1 - 1, y1 + 1) and x2 == x1 - 1 and y2 == y1 + 1
        rel promoted_bishop_diag_enemies(x, y, x2, y2, LEFT, UP) = promoted_bishop_diag_empty(x, y, x1, y1, LEFT, UP) and x1 > 0 and y1 < 8 and enemies(x1 - 1, y1 + 1) and x2 == x1 - 1 and y2 == y1 + 1
        rel promoted_bishop_diag_adjacent(x, y, x - 1, y + 1, LEFT, UP) = promoted_bishop(x, y) and enemies(x - 1, y + 1) and x > 0 and y < 8

        rel promoted_bishop_moves(x1, y1, x2, y2) = promoted_bishop_diag_empty(x1, y1, x2, y2, RIGHT, UP) or promoted_bishop_diag_enemies(x1, y1, x2, y2, RIGHT, UP) or promoted_bishop_diag_adjacent(x1, y1, x2, y2, RIGHT, UP)
        rel promoted_bishop_moves(x1, y1, x2, y2) = promoted_bishop_diag_empty(x1, y1, x2, y2, RIGHT, DOWN) or promoted_bishop_diag_enemies(x1, y1, x2, y2, RIGHT, DOWN) or promoted_bishop_diag_adjacent(x1, y1, x2, y2, RIGHT, DOWN)
        rel promoted_bishop_moves(x1, y1, x2, y2) = promoted_bishop_diag_empty(x1, y1, x2, y2, LEFT, UP) or promoted_bishop_diag_enemies(x1, y1, x2, y2, LEFT, UP) or promoted_bishop_diag_adjacent(x1, y1, x2, y2, LEFT, UP)
        rel promoted_bishop_moves(x1, y1, x2, y2) = promoted_bishop_diag_empty(x1, y1, x2, y2, LEFT, DOWN) or promoted_bishop_diag_enemies(x1, y1, x2, y2, LEFT, DOWN) or promoted_bishop_diag_adjacent(x1, y1, x2, y2, LEFT, DOWN)
        rel promoted_bishop_moves(x, y, x, y + 1) = promoted_bishop(x, y) and (empty(x, y + 1) or enemies(x, y + 1))
        rel promoted_bishop_moves(x, y, x, y - 1) = promoted_bishop(x, y) and (empty(x, y - 1) or enemies(x, y - 1))
        rel promoted_bishop_moves(x, y, x + 1, y) = promoted_bishop(x, y) and (empty(x + 1, y) or enemies(x + 1, y))
        rel promoted_bishop_moves(x, y, x - 1, y) = promoted_bishop(x, y) and (empty(x - 1, y) or enemies(x - 1, y))


        // Knight
        rel knight_moves(x, y, x - 1, y + 2) = knight(x, y) and x < 8 and y < 8 and (empty(x - 1, y + 2) or enemies(x - 1, y + 2))
        rel knight_moves(x, y, x + 1, y + 2) = knight(x, y) and x < 8 and y < 8 and (empty(x + 1, y + 2) or enemies(x + 1, y + 2))

        // Enemy_king
        rel enemy_king_moves(x, y, x - 1, y + 1) = enemy_king(x, y) and x > 0 and y < 8 and empty(x - 1, y + 1) and not check()// Up-left
        rel enemy_king_moves(x, y, x, y + 1) = enemy_king(x, y) and y < 8 and empty(x, y + 1) and not check() // Up
        rel enemy_king_moves(x, y, x + 1, y + 1) = enemy_king(x, y) and x < 8 and y < 8 and empty(x + 1, y + 1) and not check() // Up-right
        rel enemy_king_moves(x, y, x - 1, y) = enemy_king(x, y) and x > 0 and empty(x - 1, y) and not check() // Left
        rel enemy_king_moves(x, y, x + 1, y) = enemy_king(x, y) and x < 8 and empty(x + 1, y) and not check() // Right
        rel enemy_king_moves(x, y, x - 1, y - 1) = enemy_king(x, y) and x > 0 and y > 0 and empty(x - 1, y - 1) and not check() // Down-left
        rel enemy_king_moves(x, y, x, y - 1) = enemy_king(x, y) and y > 0 and empty(x, y - 1) and not check() // Down
        rel enemy_king_moves(x, y, x + 1, y - 1) = enemy_king(x, y) and x < 8 and y > 0 and empty(x + 1, y - 1) and not check() //Down-right

        // Checkmate logic
        rel num_enemy_king_moves(n) = n := count(x, y: enemy_king_moves(_, _, x, y))
        rel check() = enemy_king(x, y) and (pawn_moves(_, _, x, y) or gold_moves(_, _, x, y) or silver_moves(_, _, x, y) or lance_moves(_, _, x, y) or rook_moves(_, _, x, y) or bishop_moves(_, _, x, y) or promoted_rook_moves(_, _, x, y) or promoted_bishop_moves(_, _, x, y) or knight_moves(_, _, x, y))

        rel checkmate() = check() and num_enemy_king_moves(n) and n == 0
        query pawn
        query pawn_moves
        query lance
        query lance_moves
        query gold
        query gold_moves
        query silver
        query silver_moves
        query rook
        query rook_moves
        query promoted_rook
        query promoted_rook_moves
        query bishop
        query bishop_moves
        query promoted_bishop
        query promoted_bishop_moves
        query knight
        query knight_moves
        query enemy_king
        query enemy_king_moves
        query enemies
        query checkmate'''

setup_ctx = scallopy.Context()
pieces = ["pawn", "silver", "lance", "rook", "bishop", "knight", "promoted_pawn", "promoted_silver", "promoted_lance", "promoted_rook", "promoted_bishop", "promoted_knight", "gold", "king",
          "enemy_pawn", "enemy_silver", "enemy_lance", "enemy_rook", "enemy_bishop", "enemy_knight", "enemy_promoted_pawn", "enemy_promoted_silver", "enemy_promoted_lance", "enemy_promoted_rook", "enemy_promoted_bishop", "enemy_promoted_knight", "enemy_gold", "enemy_king"]
used_pieces = ["pawn", "gold", "silver", "lance", "rook", "promoted_rook", "bishop", "promoted_bishop", "knight", "king", "enemy_king", "enemies"]

def divideImageToGrid(image_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    cell_h, cell_w = h // 9, w // 9

    cells = []
    for row in range(9):
        for col in range(9):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            cell = img[y1:y2, x1:x2]

            # Encode to PNG bytes
            _, buffer = cv2.imencode(".png", cell)
            cell_bytes = buffer.tobytes()

            # Flip row index so (0,0) = bottom-left
            coord = (col, 8 - row)
            cells.append((coord, cell_bytes))

    return cells

def gemini_extract_pieces(image_path):
    def parse_gemini_json(response_text):
        # Try to extract JSON between ``` or ''' markers
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.S)
        if not match:
            match = re.search(r"'''(.*?)'''", response_text, re.S)

        if match:
            cleaned = match.group(1).strip()
        else:
            cleaned = response_text.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
            print("Raw response was:\n", response_text)
            return {}
        
    cells = divideImageToGrid(image_path)
    
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("models/gemini-2.0-flash-lite")

    fewshots = [
        "Classify the object in each square as the examples below.",
        "Here are examples:",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/empty/empty.png", "rb").read()},
        "empty",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/pawn/pawn.png", "rb").read()},
        "pawn",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/gold/gold.png", "rb").read()},
        "gold",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/silver/silver.png", "rb").read()},
        "silver",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/lance/lance.png", "rb").read()},
        "lance",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/rook/rook.png", "rb").read()},
        "rook",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/bishop/bishop.png", "rb").read()},
        "bishop",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/knight/knight.png", "rb").read()},
        "knight",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/king/king.png", "rb").read()},
        "king",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/promoted_pawn/promoted_pawn.png", "rb").read()},
        "promoted_pawn",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/promoted_silver/promoted_silver.png", "rb").read()},
        "promoted_silver",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/promoted_rook/promoted_rook.png", "rb").read()},
        "promoted_rook",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/promoted_bishop/promoted_bishop.png", "rb").read()},
        "promoted_bishop",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/promoted_knight/promoted_knight.png", "rb").read()},
        "promoted_knight",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/enemy_pawn/enemy_pawn.png", "rb").read()},
        "enemy_pawn",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/enemy_gold/enemy_gold.png", "rb").read()},
        "enemy_gold",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/enemy_silver/enemy_silver.png", "rb").read()},
        "enemy_silver",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/enemy_lance/enemy_lance.png", "rb").read()},
        "enemy_lance",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/enemy_rook/enemy_rook.png", "rb").read()},
        "enemy_rook",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/enemy_bishop/enemy_bishop.png", "rb").read()},
        "enemy_bishop",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/enemy_knight/enemy_knight.png", "rb").read()},
        "enemy_knight",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/enemy_king/enemy_king.png", "rb").read()},
        "enemy_king",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/enemy_promoted_pawn/enemy_promoted_pawn.png", "rb").read()},
        "enemy_promoted_pawn",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/enemy_promoted_silver/enemy_promoted_silver.png", "rb").read()},
        "enemy_promoted_silver",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/enemy_promoted_rook/enemy_promoted_rook.png", "rb").read()},
        "enemy_promoted_rook",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/enemy_promoted_bishop/enemy_promoted_bishop.png", "rb").read()},
        "enemy_promoted_bishop",
        {"mime_type": "image/png", "data": open("scallop/experiments/shogi/examples/enemy_promoted_knight/enemy_promoted_knight.png", "rb").read()},
        "enemy_promoted_knight",
        "Now classify the following 81 board squares. Return the result ONLY as JSON mapping (row,col) → class."
    ]

    request = fewshots[:]
    for (row, col), cell_bytes in cells:
        request.append(f"Square ({row},{col}):")
        request.append({"mime_type": "image/png", "data": cell_bytes})

    
    response = model.generate_content(request).text
    
    parsed = parse_gemini_json(response)

    promoted_to_gold = ["promoted_pawn", "promoted_silver", "promoted_knight", "promoted_lance"]
    enemy_promoted_to_gold = ["enemy_promoted_pawn", "enemy_promoted_silver", "enemy_promoted_knight", "enemy_promoted_lance"]

    pieces_dict = {piece: [] for piece in pieces}
    pieces_dict["enemies"] = []

    # Fill dictionary
    for pos, piece in parsed.items():
        row, col = map(int, re.findall(r"\d+", pos))
        coord = (row, col)

        # add to individual piece list
        if piece in pieces_dict:
            pieces_dict[piece].append(coord)
        
        # if it's an enemy, also add to aggregated list
        if piece.startswith("enemy_"):
            if piece != "enemy_king": 
                pieces_dict["enemies"].append(coord)

        if piece in promoted_to_gold:
            pieces_dict["gold"].append(coord)
        elif piece in enemy_promoted_to_gold:
            pieces_dict["enemy_gold"].append(coord)

    return pieces_dict

import re
from collections import defaultdict
import numpy as np
import cv2
from tensorflow.keras.models import load_model

def cnn_extract_pieces(image_path):
    cells = divideImageToGrid(image_path)
    classes = [
        "bishop", "empty", "enemy_bishop", "enemy_gold", "enemy_king",
        "enemy_knight", "enemy_lance", "enemy_pawn", "enemy_promoted_bishop",
        "enemy_promoted_knight", "enemy_promoted_lance", "enemy_promoted_pawn",
        "enemy_promoted_rook", "enemy_promoted_silver", "enemy_rook",
        "enemy_silver", "gold", "king", "knight", "lance", "pawn",
        "promoted_bishop", "promoted_knight", "promoted_lance",
        "promoted_pawn", "promoted_rook", "promoted_silver", "rook", "silver"
    ]

    promoted_to_gold = ["promoted_pawn", "promoted_silver", "promoted_knight", "promoted_lance"]
    enemy_promoted_to_gold = ["enemy_promoted_pawn", "enemy_promoted_silver", "enemy_promoted_knight", "enemy_promoted_lance"]

    # Initialize dictionary
    pieces_dict = {piece: [] for piece in classes}
    pieces_dict["enemies"] = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "shogi_cnn_model.keras")
    model = load_model(model_path)

    # Predict pieces for each cell
    for coord, cell_bytes in cells:
        # Convert bytes to image
        nparr = np.frombuffer(cell_bytes, np.uint8)
        cell_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Preprocess
        cell_img = cv2.resize(cell_img, (128, 128))
        cell_img = cell_img / 255.0
        cell_img = np.expand_dims(cell_img, axis=0)

        # Predict
        predictions = model.predict(cell_img, verbose=0)
        predicted_index = int(np.argmax(predictions))
        piece_name = classes[predicted_index]

        # Add to individual piece list
        pieces_dict[piece_name].append(coord)

        # Aggregate enemy pieces (except enemy_king)
        if piece_name.startswith("enemy_") and piece_name != "enemy_king":
            pieces_dict["enemies"].append(coord)

        # Aggregate promoted pieces as gold
        if piece_name in promoted_to_gold:
            pieces_dict["gold"].append(coord)
        elif piece_name in enemy_promoted_to_gold:
            pieces_dict["enemy_gold"].append(coord)

    return pieces_dict


def setupScallop(image_path, source):
    if source == "gemini": 
        pieces_coords = gemini_extract_pieces(image_path)
    elif source == "cnn":
        pieces_coords = cnn_extract_pieces(image_path)
    print(pieces_coords)
    setup_ctx.add_relation("pawn", (int, int))
    setup_ctx.add_relation("gold", (int, int))
    setup_ctx.add_relation("silver", (int, int))
    setup_ctx.add_relation("lance", (int, int))
    setup_ctx.add_relation("rook", (int, int))
    setup_ctx.add_relation("promoted_rook", (int, int))
    setup_ctx.add_relation("bishop", (int, int))
    setup_ctx.add_relation("promoted_bishop", (int, int))
    setup_ctx.add_relation("knight", (int, int))
    setup_ctx.add_relation("king", (int, int))
    setup_ctx.add_relation("enemy_king", (int, int))
    setup_ctx.add_relation("enemies", (int, int))
    for piece in used_pieces:
        setup_ctx.add_facts(piece, pieces_coords[piece])

    setup_ctx.add_program(program)
    setup_ctx.run()
    
def simulateMovement(piece):
    if piece == "king": return None

    piece_pos = list(setup_ctx.relation(piece))
    piece_actions = list(setup_ctx.relation(piece + "_moves"))
    piece_actions_dict = {}
    for i in range(len(piece_actions)):
        initial_pos = piece_actions[i][:2]
        next_pos = piece_actions[i][2:]
        if initial_pos not in piece_actions_dict:
            piece_actions_dict[initial_pos] = []
        piece_actions_dict[initial_pos].append(next_pos)

    for p in piece_actions_dict:
        if p not in piece_pos:
            # skip this move, or find the matching piece
            continue
        for action in piece_actions_dict[p]:
            new_pieces_pos = piece_pos.copy()
            new_pieces_pos.remove(p)
            new_pieces_pos.append(action)
            movement_ctx = scallopy.Context()
            movement_ctx.add_relation("pawn", (int, int))
            movement_ctx.add_relation("gold", (int, int))
            movement_ctx.add_relation("silver", (int, int))
            movement_ctx.add_relation("lance", (int, int))
            movement_ctx.add_relation("rook", (int, int))
            movement_ctx.add_relation("promoted_rook", (int, int))
            movement_ctx.add_relation("bishop", (int, int))
            movement_ctx.add_relation("promoted_bishop", (int, int))
            movement_ctx.add_relation("knight", (int, int))
            movement_ctx.add_relation("king", (int, int))
            movement_ctx.add_relation("enemy_king", (int, int))
            movement_ctx.add_relation("enemies", (int, int))
            
            for pc in used_pieces:
                if pc == piece:
                    movement_ctx.add_facts(piece, new_pieces_pos)
                else:
                    movement_ctx.add_facts(pc, list(setup_ctx.relation(pc)))
            
            movement_ctx.add_program(program)

            movement_ctx.run()
            ischeckmate = len(list(movement_ctx.relation("checkmate")))
            if ischeckmate: return (p, action)
    return None

def main():
    image_path = "scallop/experiments/shogi/tests/board1.png"
    
    setupScallop(image_path, "gemini")
    print("Possible solutions:")
    for piece in used_pieces[:9]:
        result = simulateMovement(piece)
        if result is not None:
            print(f"    Move {piece} from {result[0]} to {result[1]}")

main()