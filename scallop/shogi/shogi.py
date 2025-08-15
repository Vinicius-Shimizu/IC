import scallopy
import google.generativeai as genai
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for headless environments
import cv2
import numpy as np


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

        rel empty(x, y) = board(x, y) and not enemies(x, y) and not pawn(x, y) and not gold(x, y) and not silver(x, y) and not lance(x, y) and not rook(x, y) and not bishop(x, y) and not knight(x, y) and not king(x, y)


        // Pawn
        rel pawn_moves(x, y, x, y + 1) = pawn(x, y) and y < 8 and empty(x, y + 1)

        // Gold general
        rel gold_moves(x, y, x, y + 1) = gold(x, y) and y < 8 and empty(x, y + 1) 
        rel gold_moves(x, y, x - 1, y + 1) = gold(x, y) and y < 8 and x > 0 and empty(x - 1, y + 1) 
        rel gold_moves(x, y, x + 1, y + 1) = gold(x, y) and y < 8 and x < 8 and empty(x + 1, y + 1)
        rel gold_moves(x, y, x - 1, y) = gold(x, y) and x > 0 and empty(x - 1, y)
        rel gold_moves(x, y, x + 1, y) = gold(x, y) and x < 8 and empty(x + 1, y)
        rel gold_moves(x, y, x, y - 1) = gold(x, y) and y > 0 and empty(x, y - 1)

        // Silver general
        rel silver_moves(x, y, x, y + 1) = silver(x, y) and y < 8 and empty(x, y + 1)
        rel silver_moves(x, y, x - 1, y + 1) = silver(x, y) and y < 8 and x > 0 and empty(x - 1, y + 1)
        rel silver_moves(x, y, x + 1, y + 1) = silver(x, y) and y < 8 and x < 8 and empty(x + 1, y + 1)
        rel silver_moves(x, y, x - 1, y - 1) = silver(x, y) and y > 0 and x > 0 and empty(x - 1, y - 1)
        rel silver_moves(x, y, x + 1, y - 1) = silver(x, y) and y > 0 and x < 8 and empty(x + 1, y - 1)

        // Lance
        rel lance_moves(x, y, x, y + 1) = lance(x, y) and y < 8 and empty(x, y + 1)
        rel lance_moves(x, y, x, y + 1) = lance_moves(x, y - 1, x, y) and empty(x, y + 1)

        // Rook
        type rook_line(xsrc: i32, ysrc: i32, xdest: i32, ydest: i32, dir: direction)
        // UP
        rel rook_line(xsrc, ysrc, x, y + 1, UP) = rook(xsrc, ysrc) and x == xsrc and y == ysrc and y < 8 and empty(x, y + 1)
        rel rook_line(xsrc, ysrc, x, y + 1, UP) = rook_line(xsrc, ysrc, x, y, UP) and y < 8 and empty(x, y + 1)
        // DOWN
        rel rook_line(xsrc, ysrc, x, y - 1, DOWN) = rook(xsrc, ysrc) and x == xsrc and y == ysrc and y > 0 and empty(x, y - 1)
        rel rook_line(xsrc, ysrc, x, y - 1, DOWN) = rook_line(xsrc, ysrc, x, y, DOWN) and y > 0 and empty(x, y - 1)
        // RIGHT
        rel rook_line(xsrc, ysrc, x + 1, y, RIGHT) = rook(xsrc, ysrc) and x == xsrc and y == ysrc and x < 8 and empty(x + 1, y)
        rel rook_line(xsrc, ysrc, x + 1, y, RIGHT) = rook_line(xsrc, ysrc, x, y, RIGHT) and x < 8 and empty(x + 1, y)
        // LEFT
        rel rook_line(xsrc, ysrc, x - 1, y, LEFT) = rook(xsrc, ysrc) and x == xsrc and y == ysrc and x > 0 and empty(x - 1, y)
        rel rook_line(xsrc, ysrc, x - 1, y, LEFT) = rook_line(xsrc, ysrc, x, y, LEFT) and x > 0 and empty(x - 1, y)
        rel rook_moves(xsrc, ysrc, xdest, ydest) = rook_line(xsrc, ysrc, xdest, ydest, _)

        // Bishop
        type bishop_diag(xsrc: i32, ysrc: i32, xdst: i32, ydst: i32, hor_dir: direction, vert_dir: direction)
        // UP-RIGHT
        rel bishop_diag(xsrc, ysrc, x + 1, y + 1, RIGHT, UP) = bishop(xsrc, ysrc) and x == xsrc and y == ysrc and x < 8 and y < 8 and empty(x + 1, y + 1)
        rel bishop_diag(xsrc, ysrc, x + 1, y + 1, RIGHT, UP) = bishop_diag(xsrc, ysrc, x, y, RIGHT, UP) and x < 8 and y < 8 and empty(x + 1, y + 1)
        // DOWN-RIGHT
        rel bishop_diag(xsrc, ysrc, x + 1, y - 1, RIGHT, DOWN) = bishop(xsrc, ysrc) and x == xsrc and y == ysrc and x < 8 and y > 0 and empty(x + 1, y - 1)
        rel bishop_diag(xsrc, ysrc, x + 1, y - 1, RIGHT, DOWN) = bishop_diag(xsrc, ysrc, x, y, RIGHT, DOWN) and x < 8 and y > 0 and empty(x + 1, y - 1)
        // DOWN-LEFT
        rel bishop_diag(xsrc, ysrc, x - 1, y - 1, LEFT, DOWN) = bishop(xsrc, ysrc) and x == xsrc and y == ysrc and x > 0 and y > 0 and empty(x - 1, y - 1)
        rel bishop_diag(xsrc, ysrc, x - 1, y - 1, LEFT, DOWN) = bishop_diag(xsrc, ysrc, x, y, LEFT, DOWN) and x > 0 and y > 0 and empty(x - 1, y - 1)
        // UP-LEFT
        rel bishop_diag(xsrc, ysrc, x - 1, y + 1, LEFT, UP) = bishop(xsrc, ysrc) and x == xsrc and y == ysrc and x > 0 and y < 8 and empty(x - 1, y + 1)
        rel bishop_diag(xsrc, ysrc, x - 1, y + 1, LEFT, UP) = bishop_diag(xsrc, ysrc, x, y, LEFT, UP) and x > 0 and y < 8 and empty(x - 1, y + 1)
        rel bishop_moves(xsrc, ysrc, xdst, ydst) = bishop_diag(xsrc, ysrc, xdst, ydst, _, _)

        // Knight
        rel knight_moves(x, y, x - 1, y + 2) = knight(x, y) and x < 8 and y < 8 and empty(x - 1, y + 2)
        rel knight_moves(x, y, x + 1, y + 2) = knight(x, y) and x < 8 and y < 8 and empty(x + 1, y + 2)

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
        rel check() = enemy_king(x, y) and (pawn_moves(_, _, x, y) or gold_moves(_, _, x, y) or silver_moves(_, _, x, y) or lance_moves(_, _, x, y) or rook_moves(_, _, x, y) or bishop_moves(_, _, x, y) or knight_moves(_, _, x, y))

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
        query bishop
        query bishop_moves
        query knight
        query knight_moves
        query enemy_king
        query enemy_king_moves
        query enemies
        query checkmate'''
setup_ctx = scallopy.Context()
pieces = ["pawn", "silver", "lance", "rook", "bishop", "knight", "gold", "enemies", "enemy_king", "king"]



def gemini_extract_pieces(image_path):
    img = cv2.imread("test.png")
    h, w, _ = img.shape
    cell_h, cell_w = h // 9, w // 9

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("models/gemini-2.0-flash")

    chat = model.start_chat(history=[
        {
            "role": "user",
            "parts": [
                "Classify the object in the image as 'pawn', 'gold', 'silver', 'lance', 'rook', 'bishop', 'knight', 'king' and their enemy and promoted counter parts. If neither fits, classify as 'empty'.\n\nExample:",
                {"mime_type": "image/png", "data": open("examples/example_pawn.png", "rb").read()},
                "pawn"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_gold.png", "rb").read()},
                "gold"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_silver.png", "rb").read()},
                "silver"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_lance.png", "rb").read()},
                "lance"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_rook.png", "rb").read()},
                "rook"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_bishop.png", "rb").read()},
                "bishop"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_knight.png", "rb").read()},
                "knight"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_king.png", "rb").read()},
                "king"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_promoted_pawn.png", "rb").read()},
                "promoted_pawn"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_promoted_silver.png", "rb").read()},
                "promoted_silver"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_promoted_rook.png", "rb").read()},
                "promoted_rook"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_promoted_bishop.png", "rb").read()},
                "promoted_bishop"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_promoted_knight.png", "rb").read()},
                "promoted_knight"
            ]
        },
                {
            "role": "user",
            "parts": [
                "Classify the object in the image as 'pawn', 'gold', 'silver', 'lance', 'rook', 'bishop', 'knight', 'king' and their enemy and promoted counter parts.\n\nExample:",
                {"mime_type": "image/png", "data": open("examples/example_enemy_pawn.png", "rb").read()},
                "enemy_pawn"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_enemy_gold.png", "rb").read()},
                "enemy_gold"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_enemy_silver.png", "rb").read()},
                "enemy_silver"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_enemy_lance.png", "rb").read()},
                "enemy_lance"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_enemy_rook.png", "rb").read()},
                "enemy_rook"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_enemy_bishop.png", "rb").read()},
                "enemy_bishop"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_enemy_knight.png", "rb").read()},
                "enemy_knight"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_enemy_king.png", "rb").read()},
                "enemy_king"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_enemy_promoted_pawn.png", "rb").read()},
                "enemy_promoted_pawn"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_enemy_promoted_silver.png", "rb").read()},
                "enemy_promoted_silver"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_enemy_promoted_rook.png", "rb").read()},
                "enemy_promoted_rook"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_enemy_promoted_bishop.png", "rb").read()},
                "enemy_promoted_bishop"
            ]
        },
        {
            "role": "user",
            "parts": [
                {"mime_type": "image/png", "data": open("examples/example_enemy_promoted_knight.png", "rb").read()},
                "enemy_promoted_knight"
            ]
        }
    ])

    for row in range(9):
        for col in range(9):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            cell = img[y1:y2, x1:x2]

            _, buffer = cv2.imencode(".png", cell)
            cell_bytes = buffer.tobytes()
            
            response = chat.send_message([
                {"mime_type": "image/png", "data": cell_bytes}
            ])

            print(f"Cell ({row},{col}):", response.text)

def getPiecesCoords():
    pieces_coords = {}
    pieces_coords["enemies"] = [(8, 8),
            (0, 6), (6, 6), (7, 6), (8, 6),
            (1, 5), (4 ,5),
            (5, 4),
            (1, 2), (3, 2), (4, 2)]
    pieces_coords["pawn"] = [(6, 3), (0, 2), (5, 2), (8, 2)]
    pieces_coords["gold"] = [(6, 7), (5, 6), (2, 0)]
    pieces_coords["silver"] = [(6, 8), (6, 1), (5, 0)]
    pieces_coords["lance"] = [(0, 0), (8, 0)]
    pieces_coords["rook"] = []
    pieces_coords["bishop"] = []
    pieces_coords["knight"] = [(3, 3), (7, 0)]
    pieces_coords["enemy_king"] = [(8, 7)]
    pieces_coords["king"] = [(1, 0)]

    return pieces_coords

def setupScallop():
    pieces_coords = getPiecesCoords()
    setup_ctx.add_relation("pawn", (int, int))
    setup_ctx.add_relation("gold", (int, int))
    setup_ctx.add_relation("silver", (int, int))
    setup_ctx.add_relation("lance", (int, int))
    setup_ctx.add_relation("rook", (int, int))
    setup_ctx.add_relation("bishop", (int, int))
    setup_ctx.add_relation("knight", (int, int))
    setup_ctx.add_relation("king", (int, int))
    setup_ctx.add_relation("enemy_king", (int, int))
    setup_ctx.add_relation("enemies", (int, int))
    for piece in pieces:
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
            movement_ctx.add_relation("bishop", (int, int))
            movement_ctx.add_relation("knight", (int, int))
            movement_ctx.add_relation("king", (int, int))
            movement_ctx.add_relation("enemy_king", (int, int))
            movement_ctx.add_relation("enemies", (int, int))
            
            for pc in pieces:
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
    gemini_extract_pieces("test.png")
    # setupScallop()
    # for piece in pieces[:7]:
    #     result = simulateMovement(piece)
    #     if result is not None:
    #         print(f"Move {piece} from {result[0]} to {result[1]}")

main()