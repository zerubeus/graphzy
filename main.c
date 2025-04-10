#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <raylib.h> // Include Raylib header
#include <time.h>   // For seeding random number generator

#define mass 5.0f        // Much lighter for faster movement
#define stiffness 40.0f   // Stronger springs for more responsive motion
#define DeltaT 0.025f     // Larger time step for faster simulation
#define rest_length 100.0f // Shorter rest length for tighter grouping
#define damping 0.85f     // More aggressive damping for faster stabilization
#define repulsion_k 400000.0f // Increased repulsion for better node distribution
#define separation_stiffness 15000.0f // Adjusted separation force
#define node_radius 10.0f      // Visual radius of nodes

// Text embedding parameters
#define MAX_TEXT_LENGTH 256
#define MAX_WORDS 100
#define MAX_WORD_LENGTH 32
#define EMBEDDING_DIM 10
#define SIMILARITY_THRESHOLD 0.5f

// Annealing parameters
#define initial_temperature 0.8f  // Lower starting temperature
#define cooling_rate 0.98f       // Faster cooling rate
#define min_temperature 0.01f     // Minimum temperature
#define jiggle_force 120.0f       // Reduced jiggle force for less perpetual motion

// Window dimensions (can be const int or defines)
const int screenWidth = 800;
const int screenHeight = 600;

// Node structure
typedef struct Node {
	float posx;
	float posy;
	float velx; // velocity x
	float vely; // velocity y
	float accx;
	float accy;
	// Combined force components
	float forcex;
	float forcey;
	// No need for separate repulsion/force components if summed
	// float disx; // Unused
	// float repx; // Merged into forcex
	// float repy; // Merged into forcey
	char name[32]; // Increased size for word embeddings
    float embedding[EMBEDDING_DIM]; // Store embedding vector for text nodes
    int is_embedding; // Flag to indicate if this is an embedding node
} Node;

// Link structure
typedef struct Link {
	// Store indices directly for efficiency
	unsigned int start_idx;
	unsigned int end_idx;
	// Keep names if needed for other purposes, but indices are better for lookups
	// char depart[4];
	// char fin[4];
    float weight; // Similarity weight for embedding links
} Link;

// Text input state
typedef struct {
    char input_text[MAX_TEXT_LENGTH];
    bool is_editing;
    Rectangle text_box;
    bool visualization_active;
} TextInputState;

// Simple hash function for word to vector mapping
unsigned int hash_string(const char *str) {
    unsigned int hash = 5381;
    int c;
    
    while ((c = *str++))
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
        
    return hash;
}

// Generate a simple deterministic embedding for a word
void generate_word_embedding(const char *word, float *embedding) {
    unsigned int hash = hash_string(word);
    
    // Seed random generator with hash for deterministic output
    srand(hash);
    
    // Generate normalized embedding vector
    float magnitude = 0.0f;
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        embedding[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // -1 to 1
        magnitude += embedding[i] * embedding[i];
    }
    
    // Normalize to unit length
    magnitude = sqrtf(magnitude);
    if (magnitude > 0.0001f) {
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            embedding[i] /= magnitude;
        }
    }
}

// Calculate cosine similarity between two embedding vectors
float cosine_similarity(const float *vec1, const float *vec2) {
    float dot_product = 0.0f;
    float magnitude1 = 0.0f;
    float magnitude2 = 0.0f;
    
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        dot_product += vec1[i] * vec2[i];
        magnitude1 += vec1[i] * vec1[i];
        magnitude2 += vec2[i] * vec2[i];
    }
    
    magnitude1 = sqrtf(magnitude1);
    magnitude2 = sqrtf(magnitude2);
    
    if (magnitude1 < 0.0001f || magnitude2 < 0.0001f) {
        return 0.0f;
    }
    
    return dot_product / (magnitude1 * magnitude2);
}

// Create embeddings from text and add them to the graph
void create_text_embeddings(const char *text, Node **nodes, size_t *num_nodes, Link **links, size_t *num_links) {
    // Parse text into words
    char words[MAX_WORDS][MAX_WORD_LENGTH];
    int word_count = 0;
    
    // Copy text to avoid modifying original
    char text_copy[MAX_TEXT_LENGTH];
    strncpy(text_copy, text, MAX_TEXT_LENGTH - 1);
    text_copy[MAX_TEXT_LENGTH - 1] = '\0';
    
    // Split into words
    char *token = strtok(text_copy, " ,.!?;:-()\"\'");
    while (token != NULL && word_count < MAX_WORDS) {
        // Skip empty tokens
        if (strlen(token) > 0) {
            strncpy(words[word_count], token, MAX_WORD_LENGTH - 1);
            words[word_count][MAX_WORD_LENGTH - 1] = '\0';
            word_count++;
        }
        token = strtok(NULL, " ,.!?;:-()\"\'");
    }
    
    if (word_count == 0) {
        return; // No words found
    }
    
    // Allocate memory for new nodes
    size_t old_num_nodes = *num_nodes;
    size_t new_num_nodes = old_num_nodes + word_count;
    
    *nodes = realloc(*nodes, sizeof(Node) * new_num_nodes);
    if (*nodes == NULL) {
        return; // Memory allocation failed
    }
    
    // Create embedding nodes
    for (int i = 0; i < word_count; i++) {
        size_t node_idx = old_num_nodes + i;
        
        // Initialize node position randomly
        (*nodes)[node_idx].posx = (float)((rand() % (screenWidth - 200)) + 100);
        (*nodes)[node_idx].posy = (float)((rand() % (screenHeight - 200)) + 100);
        (*nodes)[node_idx].velx = 0.0f;
        (*nodes)[node_idx].vely = 0.0f;
        (*nodes)[node_idx].accx = 0.0f;
        (*nodes)[node_idx].accy = 0.0f;
        (*nodes)[node_idx].forcex = 0.0f;
        (*nodes)[node_idx].forcey = 0.0f;
        
        // Set name to the word
        strcpy((*nodes)[node_idx].name, words[i]);
        
        // Generate embedding
        generate_word_embedding(words[i], (*nodes)[node_idx].embedding);
        (*nodes)[node_idx].is_embedding = 1;
    }
    
    // Create links based on embedding similarity
    size_t max_new_links = word_count * (word_count - 1) / 2; // Maximum possible new links
    size_t old_num_links = *num_links;
    
    // Reallocate links array
    *links = realloc(*links, sizeof(Link) * (old_num_links + max_new_links));
    if (*links == NULL) {
        return; // Memory allocation failed
    }
    
    // Add links between similar embeddings
    size_t new_links = 0;
    for (int i = 0; i < word_count; i++) {
        for (int j = i + 1; j < word_count; j++) {
            float similarity = cosine_similarity(
                (*nodes)[old_num_nodes + i].embedding,
                (*nodes)[old_num_nodes + j].embedding
            );
            
            // Add link if similarity is above threshold
            if (similarity > SIMILARITY_THRESHOLD) {
                size_t link_idx = old_num_links + new_links;
                (*links)[link_idx].start_idx = old_num_nodes + i;
                (*links)[link_idx].end_idx = old_num_nodes + j;
                (*links)[link_idx].weight = similarity;
                new_links++;
            }
        }
    }
    
    // Update counts
    *num_nodes = new_num_nodes;
    *num_links = old_num_links + new_links;
}

// Perform Principal Component Analysis (PCA) to position nodes in 2D
void project_embeddings_to_2d(Node *nodes, const size_t num_nodes) {
    // Only perform if we have embedding nodes
    int has_embeddings = 0;
    for (size_t i = 0; i < num_nodes; i++) {
        if (nodes[i].is_embedding) {
            has_embeddings = 1;
            break;
        }
    }
    
    if (!has_embeddings) return;
    
    // This is a very simple approach - just use first two dimensions of embedding
    // A full PCA would require eigendecomposition which is complex
    
    // Find min/max for first two dimensions to normalize positions
    float min_x = 1000.0f, max_x = -1000.0f;
    float min_y = 1000.0f, max_y = -1000.0f;
    
    for (size_t i = 0; i < num_nodes; i++) {
        if (nodes[i].is_embedding) {
            if (nodes[i].embedding[0] < min_x) min_x = nodes[i].embedding[0];
            if (nodes[i].embedding[0] > max_x) max_x = nodes[i].embedding[0];
            if (nodes[i].embedding[1] < min_y) min_y = nodes[i].embedding[1];
            if (nodes[i].embedding[1] > max_y) max_y = nodes[i].embedding[1];
        }
    }
    
    // Position embedding nodes using first two dimensions
    float scale_x = (max_x - min_x > 0.0001f) ? (screenWidth * 0.6f) / (max_x - min_x) : 1.0f;
    float scale_y = (max_y - min_y > 0.0001f) ? (screenHeight * 0.6f) / (max_y - min_y) : 1.0f;
    
    for (size_t i = 0; i < num_nodes; i++) {
        if (nodes[i].is_embedding) {
            nodes[i].posx = (nodes[i].embedding[0] - min_x) * scale_x + screenWidth * 0.2f;
            nodes[i].posy = (nodes[i].embedding[1] - min_y) * scale_y + screenHeight * 0.2f;
        }
    }
}

// Checks if two nodes are linked based on indices
int AreNodesLinked(const Link *links, size_t num_links, unsigned int idx1, unsigned int idx2) {
    for (size_t i = 0; i < num_links; i++) {
        if ((links[i].start_idx == idx1 && links[i].end_idx == idx2) ||
            (links[i].start_idx == idx2 && links[i].end_idx == idx1)) {
            return 1;
        }
    }
    return 0;
}

// Initializes graph nodes and reads links from the file
void InitGraph(Node *nodes, const size_t num_nodes, Link *links, size_t *num_links_read, FILE *fp) {
    char s[10];
    unsigned int start_idx;
    unsigned int end_idx;
    size_t actual_links = 0;

    // Initialize node positions and properties
    for (size_t i = 0; i < num_nodes; i++) {
        nodes[i].posx = (float)((rand() % (screenWidth - 200)) + 100); // Adjust initial position based on screen size
        nodes[i].posy = (float)((rand() % (screenHeight - 200)) + 100);
        nodes[i].velx = 0.0f;
        nodes[i].vely = 0.0f;
        nodes[i].accx = 0.0f;
        nodes[i].accy = 0.0f;
        nodes[i].forcex = 0.0f;
        nodes[i].forcey = 0.0f;
        snprintf(nodes[i].name, sizeof(nodes[i].name), "%u", (unsigned int)i);
        nodes[i].is_embedding = 0; // Not an embedding node
        
        // Initialize embedding to zeros
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            nodes[i].embedding[j] = 0.0f;
        }
    }

    // Read links from the already open file pointer (fp)
    // The first line (node count) was already read in main
    while (fgets(s, sizeof(s), fp) != NULL) {
        if (sscanf(s, "%u-%u", &start_idx, &end_idx) == 2) {
            if (start_idx < num_nodes && end_idx < num_nodes && start_idx != end_idx) {
                // Avoid adding duplicate links if file contains them
                if (!AreNodesLinked(links, actual_links, start_idx, end_idx)) {
                     if (actual_links < *num_links_read) { // Check we don't exceed allocated space
                        links[actual_links].start_idx = start_idx;
                        links[actual_links].end_idx = end_idx;
                        links[actual_links].weight = 1.0f; // Default weight for regular links
                        actual_links++;
                     } else {
                         fprintf(stderr, "Warning: More links in file than counted initially. Ignoring extra links.\n");
                         break; // Stop reading links if array is full
                     }
                }
            } else {
                fprintf(stderr, "Warning: Invalid or self-referential link indices %u-%u in file.txt\n", start_idx, end_idx);
            }
        } else {
            // Optional: Warn about lines that don't match the format
            // fprintf(stderr, "Warning: Skipping malformed line in file.txt: %s", s);
        }
    }
    *num_links_read = actual_links; // Update the count to the actual number read
}

// Main physics simulation step using Velocity Verlet integration
void UpdateSimulation(Node *nodes, const size_t num_nodes, const Link *links, const size_t num_links) {
    static float temperature = initial_temperature;  // For simulated annealing
    static int frame_count = 0;
    static float energy_prev = 0.0f;
    static int stable_frames = 0;
    
    // Update temperature (cool down system gradually)
    if (temperature > min_temperature) {
        // Accelerate cooling as the graph gets larger
        float size_factor = num_nodes > 10 ? 0.99f : 1.0f;
        temperature *= cooling_rate * size_factor;
    }
    
    frame_count++;
    
    float dist, dist_sq, inv_dist;
    float deltaX, deltaY;
    float Fspring, Frepulsion;
    float forceX, forceY;
    float old_accX, old_accY;
    
    // Calculate current system energy
    float total_energy = 0.0f;
    for (size_t i = 0; i < num_nodes; i++) {
        // Kinetic energy
        float v_squared = nodes[i].velx * nodes[i].velx + nodes[i].vely * nodes[i].vely;
        total_energy += 0.5f * mass * v_squared;
    }
    
    // Check if we're stable or stuck in local minimum
    float energy_delta = fabsf(total_energy - energy_prev);
    energy_prev = total_energy;
    
    if (energy_delta < 0.1f) {
        stable_frames++;
    } else {
        stable_frames = 0;
    }
    
    // If we've been stable for a while but temperature is still high,
    // apply random perturbations to escape local minima
    bool apply_jiggle = (stable_frames > 20) && (temperature > min_temperature * 2.0f);
    
    // Only jiggle if we're not in a low-energy state already
    if (total_energy < 0.5f) {
        apply_jiggle = false;
    }
    
    // --- Velocity Verlet Step 1: Update position based on current vel and acc ---
    for (size_t i = 0; i < num_nodes; i++) {
        old_accX = nodes[i].accx; // Store current acceleration
        old_accY = nodes[i].accy;

        // pos = pos + vel*dt + 0.5*acc*dt*dt
        nodes[i].posx += nodes[i].velx * DeltaT + 0.5f * old_accX * DeltaT * DeltaT;
        nodes[i].posy += nodes[i].vely * DeltaT + 0.5f * old_accY * DeltaT * DeltaT;

        // Reset forces for recalculation at new position
        nodes[i].forcex = 0.0f;
        nodes[i].forcey = 0.0f;
        
        // Apply occasional random jiggle to escape local minima
        if (apply_jiggle) {
            float angle = ((float)rand() / RAND_MAX) * 6.28318f; // Random angle
            float jiggle_magnitude = jiggle_force * temperature;
            nodes[i].forcex += cosf(angle) * jiggle_magnitude;
            nodes[i].forcey += sinf(angle) * jiggle_magnitude;
        }
    }

    // --- Calculate Spring Forces ---
    for (size_t i = 0; i < num_links; i++) {
        unsigned int idx1 = links[i].start_idx;
        unsigned int idx2 = links[i].end_idx;

        deltaX = nodes[idx1].posx - nodes[idx2].posx;
        deltaY = nodes[idx1].posy - nodes[idx2].posy;
        dist_sq = deltaX * deltaX + deltaY * deltaY;

        if (dist_sq < 1e-4f) {
             dist = 0.01f; dist_sq = dist * dist;
             deltaX = (rand() % 2 == 0 ? 1.0f : -1.0f) * dist;
             deltaY = (rand() % 2 == 0 ? 1.0f : -1.0f) * dist;
        } else { dist = sqrtf(dist_sq); }

        // Use temperature to adjust spring stiffness
        float adaptive_stiffness = stiffness * (1.0f + temperature * 0.5f);
        
        // For embedding links, adjust rest length and stiffness based on similarity
        float link_rest_length = rest_length;
        if (nodes[links[i].start_idx].is_embedding && nodes[links[i].end_idx].is_embedding) {
            // Shorter rest length for more similar nodes
            link_rest_length = rest_length * (1.0f - links[i].weight * 0.5f);
            
            // Stronger springs for more similar nodes
            adaptive_stiffness *= (1.0f + links[i].weight);
        }
        
        Fspring = adaptive_stiffness * (dist - link_rest_length);
        inv_dist = 1.0f / dist;
        forceX = Fspring * deltaX * inv_dist;
        forceY = Fspring * deltaY * inv_dist;

        nodes[idx1].forcex -= forceX;
        nodes[idx1].forcey -= forceY;
        nodes[idx2].forcex += forceX;
        nodes[idx2].forcey += forceY;
    }

    // --- Calculate Repulsion/Separation Forces ---
    for (size_t i = 0; i < num_nodes; i++) {
        for (size_t j = i + 1; j < num_nodes; j++) {
            deltaX = nodes[i].posx - nodes[j].posx;
            deltaY = nodes[i].posy - nodes[j].posy;
            dist_sq = deltaX * deltaX + deltaY * deltaY;

            if (dist_sq < 1e-6f) { dist_sq = 1e-6f; }
            dist = sqrtf(dist_sq);

            float min_separation = 2.0f * node_radius;
            
            // Apply stronger repulsion for non-linked nodes
            float repulsion_factor = 1.0f;
            if (!AreNodesLinked(links, num_links, (unsigned int)i, (unsigned int)j)) {
                repulsion_factor = 2.5f; // Much stronger repulsion between non-linked nodes
                
                // Scale with temperature for more exploration early on
                repulsion_factor *= (1.0f + temperature);
            }

            if (dist < min_separation) {
                // Overlapping: separation force
                float overlap = min_separation - dist;
                float Fsep_magnitude = separation_stiffness * overlap * repulsion_factor;
                float inv_dist_sep = 1.0f / dist;
                float normX = deltaX * inv_dist_sep;
                float normY = deltaY * inv_dist_sep;
                forceX = Fsep_magnitude * normX;
                forceY = Fsep_magnitude * normY;
            } else {
                // Not overlapping: standard repulsion
                float inv_dist_rep = 1.0f / dist;
                Frepulsion = repulsion_k * repulsion_factor / dist_sq;
                forceX = Frepulsion * deltaX * inv_dist_rep;
                forceY = Frepulsion * deltaY * inv_dist_rep;
            }
            nodes[i].forcex += forceX;
            nodes[i].forcey += forceY;
            nodes[j].forcex -= forceX;
            nodes[j].forcey -= forceY;
        }
    }

    // --- Velocity Verlet Step 2 & 3: Update acceleration and velocity ---
    float max_velocity = 0.0f;
    for (size_t i = 0; i < num_nodes; i++) {
        // Calculate NEW acceleration: a_new = F_new / m
        float new_accX = nodes[i].forcex / mass;
        float new_accY = nodes[i].forcey / mass;

        // Update velocity: v = v + 0.5*(a_old + a_new)*dt
        nodes[i].velx += 0.5f * (old_accX + new_accX) * DeltaT;
        nodes[i].vely += 0.5f * (old_accY + new_accY) * DeltaT;

        // Use temperature-dependent damping (less damping when hot)
        float adaptive_damping = damping + (1.0f - damping) * (1.0f - temperature);
        
        // For large graphs, increase damping even more as temperature drops
        if (num_nodes > 10 && temperature < 0.3f) {
            adaptive_damping *= 0.97f;
        }
        
        nodes[i].velx *= adaptive_damping;
        nodes[i].vely *= adaptive_damping;
        
        // Apply extra strong damping when almost stable
        if (temperature < 0.1f) {
            nodes[i].velx *= 0.95f;
            nodes[i].vely *= 0.95f;
        }
        
        // Track maximum velocity for detecting stability
        float vel_sq = nodes[i].velx * nodes[i].velx + nodes[i].vely * nodes[i].vely;
        if (vel_sq > max_velocity) {
            max_velocity = vel_sq;
        }

        // Store new acceleration for next frame's step 1
        nodes[i].accx = new_accX;
        nodes[i].accy = new_accY;

        // Boundary checks (remain the same)
        if (nodes[i].posx < node_radius) { nodes[i].posx = node_radius; nodes[i].velx *= -0.5f; }
        if (nodes[i].posx > screenWidth - node_radius) { nodes[i].posx = screenWidth - node_radius; nodes[i].velx *= -0.5f; }
        if (nodes[i].posy < node_radius) { nodes[i].posy = node_radius; nodes[i].vely *= -0.5f; }
        if (nodes[i].posy > screenHeight - node_radius) { nodes[i].posy = screenHeight - node_radius; nodes[i].vely *= -0.5f; }
    }

    // --- Centering Step: Keep the graph centered on screen ---
    if (num_nodes > 0) { // Avoid division by zero if no nodes
        float sumX = 0.0f;
        float sumY = 0.0f;
        for (size_t i = 0; i < num_nodes; i++) {
            sumX += nodes[i].posx;
            sumY += nodes[i].posy;
        }
        float avgX = sumX / (float)num_nodes;
        float avgY = sumY / (float)num_nodes;

        float targetX = screenWidth / 2.0f;
        float targetY = screenHeight / 2.0f;

        float offsetX = targetX - avgX;
        float offsetY = targetY - avgY;

        // Apply the offset to all nodes
        for (size_t i = 0; i < num_nodes; i++) {
            nodes[i].posx += offsetX;
            nodes[i].posy += offsetY;
        }
    }
}

// Renders the graph using Raylib functions
void RenderGraph(const Node *graph_nodes, const size_t num_nodes, const Link *links, const size_t num_links) {
    // Define colors using Raylib COLOR types
    Color nodeColor = WHITE;
    Color embeddingNodeColor = GOLD;
    Color nodeOutlineColor = GRAY;
    Color lineColor = BLUE;
    Color embeddingLineColor = GREEN;
    Color textColor = BLACK;
    int fontSize = 10; // Font size for node labels

    BeginDrawing();
    ClearBackground(BLACK); // Clear with black background

    // Draw lines first
    for (size_t i = 0; i < num_links; i++) {
        unsigned int idx1 = links[i].start_idx;
        unsigned int idx2 = links[i].end_idx;
        // Create Vector2 for line endpoints
        Vector2 startPos = { graph_nodes[idx1].posx, graph_nodes[idx1].posy };
        Vector2 endPos = { graph_nodes[idx2].posx, graph_nodes[idx2].posy };
        
        // Use different color for embedding links
        Color currentLineColor = lineColor;
        if (graph_nodes[idx1].is_embedding && graph_nodes[idx2].is_embedding) {
            // Adjust color alpha based on similarity
            currentLineColor = embeddingLineColor;
            
            // Adjust line thickness based on similarity
            float thickness = 1.0f + links[i].weight * 2.0f;
            DrawLineEx(startPos, endPos, thickness, currentLineColor);
        } else {
            DrawLineV(startPos, endPos, currentLineColor);
        }
    }

    // Draw nodes and text
    for (size_t i = 0; i < num_nodes; i++) {
        // Create Vector2 for circle center
        Vector2 center = { graph_nodes[i].posx, graph_nodes[i].posy };
        
        // Use different color for embedding nodes
        Color currentNodeColor = graph_nodes[i].is_embedding ? embeddingNodeColor : nodeColor;

        // Draw circle outline first then filled circle
        DrawCircleV(center, node_radius + 1, nodeOutlineColor); // Slightly larger for outline
        DrawCircleV(center, node_radius, currentNodeColor);

        // Draw text (centered on node)
        // Measure text to center it
        int textWidth = MeasureText(graph_nodes[i].name, fontSize);
        Vector2 textPos = {
            center.x - (float)textWidth / 2.0f,
            center.y - (float)fontSize / 2.0f
        };
        DrawText(graph_nodes[i].name, (int)textPos.x, (int)textPos.y, fontSize, textColor);
    }

    EndDrawing();
}

// Render the text input UI
void RenderTextInput(TextInputState *state) {
    if (!state->is_editing) return;
    
    // Draw semi-transparent background
    DrawRectangle(0, 0, screenWidth, screenHeight, (Color){0, 0, 0, 100});
    
    // Draw input box
    DrawRectangleRec(state->text_box, WHITE);
    DrawRectangleLinesEx(state->text_box, 2, BLUE);
    
    // Draw input text
    int fontSize = 20;
    DrawText(state->input_text, state->text_box.x + 10, state->text_box.y + 10, fontSize, BLACK);
    
    // Draw cursor for text
    if ((GetTime() * 2) - (int)(GetTime() * 2) < 1) {
        int textWidth = MeasureText(state->input_text, fontSize);
        DrawLine(
            state->text_box.x + 10 + textWidth, 
            state->text_box.y + 8, 
            state->text_box.x + 10 + textWidth, 
            state->text_box.y + state->text_box.height - 8, 
            BLACK
        );
    }
    
    // Draw instructions
    DrawText("Type your text for embedding visualization.", 10, 10, 20, WHITE);
    DrawText("Press ENTER to visualize, ESC to cancel", 10, 40, 20, WHITE);
}

// Handle text input
bool HandleTextInput(TextInputState *state, Node **nodes, size_t *num_nodes, Link **links, size_t *num_links) {
    if (!state->is_editing) {
        // Check for T key to start text input
        if (IsKeyPressed(KEY_T)) {
            state->is_editing = true;
            state->input_text[0] = '\0';
            state->text_box = (Rectangle){
                screenWidth / 2.0f - 200,
                screenHeight / 2.0f - 30,
                400,
                60
            };
        }
        return false;
    }
    
    // Handle ESC to cancel
    if (IsKeyPressed(KEY_ESCAPE)) {
        state->is_editing = false;
        return false;
    }
    
    // Handle ENTER to submit
    if (IsKeyPressed(KEY_ENTER)) {
        state->is_editing = false;
        
        // Only create embeddings if text is not empty
        if (strlen(state->input_text) > 0) {
            create_text_embeddings(state->input_text, nodes, num_nodes, links, num_links);
            project_embeddings_to_2d(*nodes, *num_nodes);
            state->visualization_active = true;
            return true;
        }
        return false;
    }
    
    // Handle backspace
    if (IsKeyPressed(KEY_BACKSPACE)) {
        int textLen = strlen(state->input_text);
        if (textLen > 0) {
            state->input_text[textLen - 1] = '\0';
        }
    }
    
    // Handle text input
    int key = GetCharPressed();
    while (key > 0) {
        int textLen = strlen(state->input_text);
        
        // Add character if not too long
        if (textLen < MAX_TEXT_LENGTH - 1) {
            state->input_text[textLen] = (char)key;
            state->input_text[textLen + 1] = '\0';
        }
        
        key = GetCharPressed();  // Check next character in queue
    }
    
    return false;
}

int main(int argc, char *argv[]) {
    FILE *fp = NULL;
    size_t num_nodes = 0;
    size_t num_links_alloc = 0;
    size_t num_links_actual = 0;
    char s[20];
    char *filename = "file.txt"; // Default filename

    // Check if a filename was provided as command-line argument
    if (argc > 1) {
        filename = argv[1];
    }

    // Seed random number generator
    srand((unsigned int)time(NULL)); // Requires #include <time.h>

    // --- File Reading and Graph Initialization ---
    fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return EXIT_FAILURE;
    }

    // Read number of nodes directly into size_t using %zu
    if (fgets(s, sizeof(s), fp) == NULL || sscanf(s, "%zu", &num_nodes) != 1 || num_nodes == 0 || num_nodes > 10000) { // Use %zu and read directly into num_nodes
        fprintf(stderr, "Error reading a valid number of nodes (1-10000) from %s\n", filename);
        fclose(fp);
        return EXIT_FAILURE;
    }

    // Count number of potential links lines for allocation
    while (fgets(s, sizeof(s), fp) != NULL) {
        if (strchr(s, '-')) { // Simple check for a potential link line
            num_links_alloc++;
        }
    }
     if (num_links_alloc == 0) {
        fprintf(stderr,"Warning: No link lines found in %s\n", filename);
     }

    rewind(fp);             // Go back to the beginning
    fgets(s, sizeof(s), fp); // Skip the first line (number of nodes) again

    // Allocate memory
    Node *nodes = malloc(sizeof(Node) * num_nodes);
    if (nodes == NULL) {
        perror("Failed to allocate memory for nodes");
        fclose(fp);
        return EXIT_FAILURE;
    }
    // Allocate based on counted lines, even if some might be invalid
    Link *links = NULL;
     if (num_links_alloc > 0) {
        links = malloc(sizeof(Link) * num_links_alloc);
         if (links == NULL) {
            perror("Failed to allocate memory for links");
            free(nodes);
            fclose(fp);
            return EXIT_FAILURE;
        }
     } else {
         links = NULL; // No links to allocate
     }


    // Initialize graph (reads links and updates num_links_actual)
    num_links_actual = num_links_alloc; // Pass the allocated size
    InitGraph(nodes, num_nodes, links, &num_links_actual, fp);
    fclose(fp); // Close file after InitGraph finishes reading it

     if (num_links_actual != num_links_alloc) {
         fprintf(stderr, "Info: Allocated for %zu links, but read %zu valid links.\n", num_links_alloc, num_links_actual);
         // Optional: realloc 'links' to 'num_links_actual' if memory is critical
     }

    // Initialize text input state
    TextInputState input_state = {
        .input_text = "",
        .is_editing = false,
        .visualization_active = false
    };

    // --- Raylib Setup ---
    char window_title[100];
    snprintf(window_title, sizeof(window_title), "Graph with Text Embeddings - %s", filename);
    InitWindow(screenWidth, screenHeight, window_title);
    SetTargetFPS(60); // Set desired frame rate

    // --- Main Loop ---
    // Use Raylib's WindowShouldClose() function
    while (!WindowShouldClose()) {
        // Handle text input and embeddings creation
        if (HandleTextInput(&input_state, &nodes, &num_nodes, &links, &num_links_actual)) {
            // Reset simulation temperature when new embeddings are added
            // This is done implicitly in UpdateSimulation as it uses a static temperature
        }
        
        // Simulation step
        UpdateSimulation(nodes, num_nodes, links, num_links_actual);

        // Rendering step
        RenderGraph(nodes, num_nodes, links, num_links_actual);
        
        // Render text input UI if active
        RenderTextInput(&input_state);
        
        // Display help text
        if (!input_state.is_editing) {
            DrawText("Press 'T' to enter text for embedding visualization", 10, 10, 20, WHITE);
        }
    }

    // --- Cleanup ---
    CloseWindow(); // Close Raylib window and OpenGL context
    free(nodes);
    if (links) free(links); // Free links only if allocated

    return EXIT_SUCCESS;
} 