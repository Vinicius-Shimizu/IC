import scallopy

ctx = scallopy.Context()
def setupScallop():
    ctx.add_program('''
            // enemies
            rel enemies = {
                (8, 8),
                (0, 6), (6, 6), (7, 6), (8, 6),
                (1, 5), (4 ,5),
                (5, 4),
                (1, 2), (3, 2), (4, 2)
            }

            type direction = UP | DOWN | LEFT | RIGHT

            type board(x: usize, y: usize)
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

            rel empty(x, y) = board(x, y) and not enemies(x, y) and not pawn(x, y) and not gold(x, y) and not silver(x, y) and not lance(x, y) and not rook(x, y) and not bishop(x, y) and not knight(x, y)


            // Pawn
            rel pawn = {
                (6, 3), (0, 2), (5, 2), (8, 2)
            }
            rel pawn_moves(x, y + 1) = pawn(x, y) and y < 8 and empty(x, y + 1)

            // Gold general
            type gold(x: usize, y: usize)
            rel gold = {
                (7, 7), (5, 6), (2, 0)
            }
            rel gold_moves(x, y + 1) = gold(x, y) and y < 8 and empty(x, y + 1) 
            rel gold_moves(x - 1, y + 1) = gold(x, y) and y < 8 and x > 0 and empty(x - 1, y + 1) 
            rel gold_moves(x + 1, y + 1) = gold(x, y) and y < 8 and x < 8 and empty(x + 1, y + 1)
            rel gold_moves(x - 1, y) = gold(x, y) and x > 0 and empty(x - 1, y)
            rel gold_moves(x + 1, y) = gold(x, y) and x < 8 and empty(x + 1, y)
            rel gold_moves(x, y - 1) = gold(x, y) and y > 0 and empty(x, y - 1)

            // Silver general
            type silver(x: usize, y:usize)
            rel silver = {
                (6, 8), (6, 1), (5, 0)
            }
            rel silver_moves(x, y + 1) = silver(x, y) and y < 8 and empty(x, y + 1)
            rel silver_moves(x - 1, y + 1) = silver(x, y) and y < 8 and x > 0 and empty(x - 1, y + 1)
            rel silver_moves(x + 1, y + 1) = silver(x, y) and y < 8 and x < 8 and empty(x + 1, y + 1)
            rel silver_moves(x - 1, y - 1) = silver(x, y) and y > 0 and x > 0 and empty(x - 1, y - 1)
            rel silver_moves(x + 1, y - 1) = silver(x, y) and y > 0 and x < 8 and empty(x + 1, y - 1)

            // Lance
            type lance(x: usize, y:usize)
            rel lance = {
                (0, 0), (8, 0)
            }
            rel lance_moves(x, y + 1) = lance(x, y) and y < 8 and empty(x, y + 1)
            rel lance_moves(x, y + 1) = lance_moves(x, y) and empty(x, y + 1)

            // Rook
            type rook(x: usize, y:usize)
            rel rook = {
                
            }
            type rook_line(x: usize, y : usize, dir: direction)
            rel rook_line(x, y + 1, UP) = rook(x, y) and y < 8 and empty(x, y + 1)
            rel rook_line(x, y + 1, UP) = rook_line(x, y, UP) and y < 8 and empty(x, y + 1)
            rel rook_line(x, y - 1, DOWN) = rook(x, y) and y > 0 and empty(x, y - 1)
            rel rook_line(x, y - 1, DOWN) = rook_line(x, y, DOWN) and y > 0 and empty(x, y - 1)
            rel rook_line(x + 1, y, RIGHT) = rook(x, y) and x < 8 and empty(x + 1, y)
            rel rook_line(x + 1, y, RIGHT) = rook_line(x, y, RIGHT) and x < 8 and empty(x + 1, y)
            rel rook_line(x - 1, y, LEFT) = rook(x, y) and x > 0 and empty(x - 1, y)
            rel rook_line(x - 1, y, LEFT) = rook_line(x, y, LEFT) and x > 0 and empty(x - 1, y)
            rel rook_moves(x, y) = rook_line(x, y, _)

            // Bishop
            type bishop(x: usize, y:usize)
            rel bishop = {
                
            }
            type bishop_diag(x: usize, y: usize, hor_dir: direction, vert_dir: direction)
            rel bishop_diag(x + 1, y + 1, UP, RIGHT) = bishop(x, y) and x < 8 and y < 8 and empty(x + 1, y + 1)
            rel bishop_diag(x + 1, y + 1, UP, RIGHT) = bishop_diag(x, y, UP, RIGHT) and x < 8 and y < 8 and empty(x + 1, y + 1)
            rel bishop_diag(x + 1, y - 1, DOWN, RIGHT) = bishop(x, y) and x < 8 and y > 0 and empty(x + 1, y - 1)
            rel bishop_diag(x + 1, y - 1, DOWN, RIGHT) = bishop_diag(x, y, DOWN, RIGHT) and x < 8 and y > 0 and empty(x + 1, y - 1)
            rel bishop_diag(x - 1, y - 1, DOWN, LEFT) = bishop(x, y) and x > 0 and y > 0 and empty(x - 1, y - 1)
            rel bishop_diag(x - 1, y - 1, DOWN, LEFT) = bishop_diag(x, y, DOWN, LEFT) and x > 0 and y > 0 and empty(x - 1, y - 1)
            rel bishop_diag(x - 1, y + 1, UP, LEFT) = bishop(x, y) and x > 0 and y < 8 and empty(x - 1, y + 1)
            rel bishop_diag(x - 1, y + 1, UP, LEFT) = bishop_diag(x, y, UP, LEFT) and x > 0 and y < 8 and empty(x - 1, y + 1)
            rel bishop_moves(x, y) = bishop_diag(x, y, _, _)

            // Knight
            type knight(x: usize, y:usize)
            rel knight = {
                (3, 3), (7, 0)
            }
            rel knight_moves(x - 1, y + 2) = knight(x, y) and x < 8 and y < 8 and empty(x - 1, y + 2)
            rel knight_moves(x + 1, y + 2) = knight(x, y) and x < 8 and y < 8 and empty(x + 1, y + 2)

            // Enemy_king
            type enemy_king(x: usize, y:usize)
            rel enemy_king = {
                (8, 7)
            }
            rel enemy_king_moves(x - 1, y + 1) = enemy_king(x, y) and x > 0 and y < 8 and empty(x - 1, y + 1) and not check()// Up-left
            rel enemy_king_moves(x, y + 1) = enemy_king(x, y) and y < 8 and empty(x, y + 1) and not check() // Up
            rel enemy_king_moves(x + 1, y + 1) = enemy_king(x, y) and x < 8 and y < 8 and empty(x + 1, y + 1) and not check() // Up-right
            rel enemy_king_moves(x - 1, y) = enemy_king(x, y) and x > 0 and empty(x - 1, y) and not check() // Left
            rel enemy_king_moves(x + 1, y) = enemy_king(x, y) and x < 8 and empty(x + 1, y) and not check() // Right
            rel enemy_king_moves(x - 1, y - 1) = enemy_king(x, y) and x > 0 and y > 0 and empty(x - 1, y - 1) and not check() // Down-left
            rel enemy_king_moves(x, y - 1) = enemy_king(x, y) and y > 0 and empty(x, y - 1) and not check() // Down
            rel enemy_king_moves(x + 1, y - 1) = enemy_king(x, y) and x < 8 and y > 0 and empty(x + 1, y - 1) and not check() //Down-right


            // Checkmate logic
            rel num_enemy_king_moves(n) = n := count(x, y: enemy_king_moves(x, y))
            rel check() = enemy_king(x, y) and gold_moves(x, y)

            rel checkmate() = check() and num_enemy_king_moves(n) and n == 0
            query pawn_moves
            query checkmate''')
    ctx.run()
    
def main():    
    setupScallop()
    ischeckmate = len(list(ctx.relation("checkmate"))) 
    if ischeckmate: print("CHECKMATE!!")
    else: print("Failed...")
    

main()