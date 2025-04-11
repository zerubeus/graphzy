#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <raylib.h> // Include Raylib header
#include <time.h>   // For seeding random number generator
#include <ctype.h>  // For tolower function
#include <stdbool.h> // For bool type

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
// Contextual embedding dictionary size
#define DICTIONARY_SIZE 50
#define CONTEXT_WINDOW 5  // For word co-occurrence analysis

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

// Dictionary entry for semantic relationships
typedef struct {
    char word[MAX_WORD_LENGTH];
    float embedding[EMBEDDING_DIM];
    // Related words for king-queen type relationships
    char related_words[3][MAX_WORD_LENGTH];
    float relation_vectors[3][EMBEDDING_DIM];
} DictionaryEntry;

// Global dictionary with predefined semantic relationships
DictionaryEntry semantic_dictionary[DICTIONARY_SIZE] = {0};

// Initialize the semantic dictionary with common words and their relationships
void initialize_semantic_dictionary() {
    // Common words with semantic relationships
    const char* word_pairs[][2] = {
        {"man", "woman"}, {"king", "queen"}, {"brother", "sister"},
        {"boy", "girl"}, {"father", "mother"}, {"son", "daughter"},
        {"he", "she"}, {"him", "her"}, {"uncle", "aunt"},
        {"actor", "actress"}, {"waiter", "waitress"}, {"husband", "wife"}
    };
    
    // Common categories
    const char* categories[][4] = {
        {"red", "blue", "green", "yellow"},      // colors
        {"dog", "cat", "bird", "fish"},          // animals
        {"happy", "sad", "angry", "excited"},    // emotions
        {"car", "bus", "train", "plane"},        // vehicles
        {"north", "south", "east", "west"}       // directions
    };
    
    // Abstract concepts
    const char* abstract_concepts[] = {
        "time", "space", "love", "hate", "truth", "lie", "good", "bad", 
        "beauty", "ugly", "knowledge", "ignorance", "hope", "despair"
    };
    
    // Initialize base dictionary with random but consistent embeddings
    int dict_index = 0;
    
    // Add gender pairs with related embeddings
    for (int i = 0; i < 12 && dict_index < DICTIONARY_SIZE - 1; i++) {
        // First word in pair
        strncpy(semantic_dictionary[dict_index].word, word_pairs[i][0], MAX_WORD_LENGTH-1);
        
        // Generate base embedding deterministically
        unsigned int seed = hash_string(word_pairs[i][0]);
        srand(seed);
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            semantic_dictionary[dict_index].embedding[j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
        
        // Add related word information
        strncpy(semantic_dictionary[dict_index].related_words[0], word_pairs[i][1], MAX_WORD_LENGTH-1);
        
        dict_index++;
        
        // Second word in pair (with relationship to first)
        strncpy(semantic_dictionary[dict_index].word, word_pairs[i][1], MAX_WORD_LENGTH-1);
        
        // Create embedding based on first word - ensure they're related
        // This simulates the king - man + woman = queen type relationships
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            float random_offset = ((float)rand() / RAND_MAX) * 0.3f - 0.15f;
            semantic_dictionary[dict_index].embedding[j] = semantic_dictionary[dict_index-1].embedding[j] + random_offset;
        }
        
        // Add related word information
        strncpy(semantic_dictionary[dict_index].related_words[0], word_pairs[i][0], MAX_WORD_LENGTH-1);
        
        dict_index++;
    }
    
    // Add category words with similar embeddings within category
    for (int i = 0; i < 5 && dict_index < DICTIONARY_SIZE - 4; i++) {
        float category_base[EMBEDDING_DIM] = {0};
        
        // Generate base vector for category
        unsigned int seed = hash_string(categories[i][0]) ^ hash_string(categories[i][1]);
        srand(seed);
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            category_base[j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
        
        // Create entries for each word in category with similar embeddings
        for (int k = 0; k < 4 && dict_index < DICTIONARY_SIZE; k++) {
            strncpy(semantic_dictionary[dict_index].word, categories[i][k], MAX_WORD_LENGTH-1);
            
            // Generate embedding with small deviation from category base
            srand(hash_string(categories[i][k]));
            for (int j = 0; j < EMBEDDING_DIM; j++) {
                float random_offset = ((float)rand() / RAND_MAX) * 0.4f - 0.2f;
                semantic_dictionary[dict_index].embedding[j] = category_base[j] + random_offset;
            }
            
            // Store related words (others in category)
            int rel_idx = 0;
            for (int m = 0; m < 4; m++) {
                if (m != k && rel_idx < 3) {
                    strncpy(semantic_dictionary[dict_index].related_words[rel_idx], 
                            categories[i][m], MAX_WORD_LENGTH-1);
                    rel_idx++;
                }
            }
            
            dict_index++;
        }
    }
    
    // Normalize all embeddings to unit length
    for (int i = 0; i < dict_index; i++) {
        float magnitude = 0.0f;
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            magnitude += semantic_dictionary[i].embedding[j] * semantic_dictionary[i].embedding[j];
        }
        
        magnitude = sqrtf(magnitude);
        if (magnitude > 0.0001f) {
            for (int j = 0; j < EMBEDDING_DIM; j++) {
                semantic_dictionary[i].embedding[j] /= magnitude;
            }
        }
    }
}

// Find word in the semantic dictionary
int find_in_dictionary(const char *word) {
    for (int i = 0; i < DICTIONARY_SIZE; i++) {
        if (strcmp(semantic_dictionary[i].word, word) == 0) {
            return i;
        }
    }
    return -1; // Not found
}

// Generate contextual embedding based on surrounding words
void generate_contextual_embedding(const char *word, const char **context, int context_size, float *embedding) {
    // Check if word is in dictionary
    int word_idx = find_in_dictionary(word);
    
    if (word_idx >= 0) {
        // Start with dictionary embedding
        memcpy(embedding, semantic_dictionary[word_idx].embedding, EMBEDDING_DIM * sizeof(float));
    } else {
        // Generate deterministic but random embedding for unknown words
        unsigned int seed = hash_string(word);
        srand(seed);
        
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            embedding[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
    
    // Influence embedding based on context (surrounding words)
    if (context_size > 0) {
        float context_influence = 0.2f; // How much context affects the embedding
        
        for (int i = 0; i < context_size; i++) {
            int context_idx = find_in_dictionary(context[i]);
            
            if (context_idx >= 0) {
                // Add weighted influence from dictionary words in context
                for (int j = 0; j < EMBEDDING_DIM; j++) {
                    embedding[j] += semantic_dictionary[context_idx].embedding[j] * 
                                    context_influence / (float)context_size;
                }
            }
        }
        
        // Re-normalize to unit length
        float magnitude = 0.0f;
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            magnitude += embedding[i] * embedding[i];
        }
        
        magnitude = sqrtf(magnitude);
        if (magnitude > 0.0001f) {
            for (int i = 0; i < EMBEDDING_DIM; i++) {
                embedding[i] /= magnitude;
            }
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

// Generate word embedding considering semantic meaning and context
void generate_word_embedding(const char *word, float *embedding, const char **context, int context_size) {
    // Convert word to lowercase for better matching
    char word_lower[MAX_WORD_LENGTH];
    strncpy(word_lower, word, MAX_WORD_LENGTH - 1);
    word_lower[MAX_WORD_LENGTH - 1] = '\0';
    
    for (int i = 0; word_lower[i]; i++) {
        word_lower[i] = tolower(word_lower[i]);
    }
    
    // Generate contextual embedding
    generate_contextual_embedding(word_lower, context, context_size, embedding);
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
            // Convert to lowercase for better matching
            char word_lower[MAX_WORD_LENGTH];
            strncpy(word_lower, token, MAX_WORD_LENGTH - 1);
            word_lower[MAX_WORD_LENGTH - 1] = '\0';
            
            for (int i = 0; word_lower[i]; i++) {
                word_lower[i] = tolower(word_lower[i]);
            }
            
            strncpy(words[word_count], word_lower, MAX_WORD_LENGTH - 1);
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
    
    // Create embedding nodes with contextual information
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
        
        // Create context window for this word
        const char *context[CONTEXT_WINDOW * 2];
        int context_size = 0;
        
        // Look at words before and after current word (within context window)
        for (int j = i - CONTEXT_WINDOW; j <= i + CONTEXT_WINDOW; j++) {
            if (j >= 0 && j < word_count && j != i) {
                context[context_size++] = words[j];
            }
        }
        
        // Generate embedding with context
        generate_word_embedding(words[i], (*nodes)[node_idx].embedding, context, context_size);
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
            
            // Always create links between semantically related words from the dictionary
            else {
                // Check if words have a semantic relationship in the dictionary
                int idx1 = find_in_dictionary((*nodes)[old_num_nodes + i].name);
                int idx2 = find_in_dictionary((*nodes)[old_num_nodes + j].name);
                
                bool is_related = false;
                
                // Check if either word lists the other as related
                if (idx1 >= 0) {
                    for (int k = 0; k < 3; k++) {
                        if (semantic_dictionary[idx1].related_words[k][0] != '\0' && 
                            strcmp(semantic_dictionary[idx1].related_words[k], 
                                   (*nodes)[old_num_nodes + j].name) == 0) {
                            is_related = true;
                            break;
                        }
                    }
                }
                
                if (!is_related && idx2 >= 0) {
                    for (int k = 0; k < 3; k++) {
                        if (semantic_dictionary[idx2].related_words[k][0] != '\0' && 
                            strcmp(semantic_dictionary[idx2].related_words[k], 
                                   (*nodes)[old_num_nodes + i].name) == 0) {
                            is_related = true;
                            break;
                        }
                    }
                }
                
                // If they're related in the dictionary, connect them
                if (is_related) {
                    size_t link_idx = old_num_links + new_links;
                    (*links)[link_idx].start_idx = old_num_nodes + i;
                    (*links)[link_idx].end_idx = old_num_nodes + j;
                    (*links)[link_idx].weight = 0.7f; // Strong weight for semantic pairs
                    new_links++;
                }
                // Check if they belong to the same category using a lower similarity threshold
                else if (similarity > 0.25f) {
                    size_t link_idx = old_num_links + new_links;
                    (*links)[link_idx].start_idx = old_num_nodes + i;
                    (*links)[link_idx].end_idx = old_num_nodes + j;
                    (*links)[link_idx].weight = similarity;
                    new_links++;
                }
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
    Color semanticLineColor = RED;     // For strongly related semantic words
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
            // Check if it's a strong semantic relationship
            if (links[i].weight > 0.6f) {
                currentLineColor = semanticLineColor;
                DrawLineEx(startPos, endPos, 2.0f + links[i].weight * 1.5f, currentLineColor);
            } else {
                // Adjust color based on similarity
                currentLineColor = embeddingLineColor;
                float alpha = 150 + (links[i].weight * 100); // Transparency based on strength
                if (alpha > 255) alpha = 255;
                currentLineColor.a = (unsigned char)alpha;
                
                // Adjust line thickness based on similarity
                float thickness = 0.5f + links[i].weight * 3.0f;
                DrawLineEx(startPos, endPos, thickness, currentLineColor);
            }
        } else {
            DrawLineV(startPos, endPos, currentLineColor);
        }
    }

    // Draw nodes and text
    for (size_t i = 0; i < num_nodes; i++) {
        // Create Vector2 for circle center
        Vector2 center = { graph_nodes[i].posx, graph_nodes[i].posy };
        
        // Use different color for embedding nodes
        Color currentNodeColor = nodeColor;
        float scale = 1.0f;
        
        if (graph_nodes[i].is_embedding) {
            // Check if it's a dictionary word
            if (find_in_dictionary(graph_nodes[i].name) >= 0) {
                currentNodeColor = RED;   // Dictionary words highlighted
                scale = 1.2f;             // Slightly larger
            } else {
                currentNodeColor = embeddingNodeColor;
            }
        }

        // Draw circle outline first then filled circle
        DrawCircleV(center, node_radius * scale + 1, nodeOutlineColor); // Slightly larger for outline
        DrawCircleV(center, node_radius * scale, currentNodeColor);

        // Draw text (centered on node)
        // Measure text to center it
        int textWidth = MeasureText(graph_nodes[i].name, fontSize);
        Vector2 textPos = {
            center.x - (float)textWidth / 2.0f,
            center.y - (float)fontSize / 2.0f
        };
        DrawText(graph_nodes[i].name, (int)textPos.x, (int)textPos.y, fontSize, textColor);
    }

    // Display dictionary info
    DrawText("Dictionary-based semantic embeddings active", 10, screenHeight - 30, 15, GREEN);

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
    // Initialize with no nodes or links
    size_t num_nodes = 0;
    size_t num_links = 0;
    
    // Allocate initial empty arrays
    Node *nodes = malloc(sizeof(Node) * 1); // Start with space for at least 1 node
    Link *links = malloc(sizeof(Link) * 1); // Start with space for at least 1 link
    
    if (nodes == NULL || links == NULL) {
        fprintf(stderr, "Failed to allocate initial memory\n");
        if (nodes) free(nodes);
        if (links) free(links);
        return EXIT_FAILURE;
    }

    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Initialize text input state
    TextInputState input_state = {
        .input_text = "",
        .is_editing = false,
        .visualization_active = false
    };
    
    // Initialize semantic dictionary
    initialize_semantic_dictionary();

    // --- Raylib Setup ---
    char window_title[100];
    snprintf(window_title, sizeof(window_title), "Semantic Embedding Visualization");
    InitWindow(screenWidth, screenHeight, window_title);
    SetTargetFPS(60); // Set desired frame rate

    // --- Main Loop ---
    while (!WindowShouldClose()) {
        // Handle text input and embeddings creation
        if (HandleTextInput(&input_state, &nodes, &num_nodes, &links, &num_links)) {
            // Reset simulation temperature when new embeddings are added
            // This is done implicitly in UpdateSimulation as it uses a static temperature
        }
        
        // Only run simulation and render if we have nodes
        if (num_nodes > 0) {
            // Simulation step
            UpdateSimulation(nodes, num_nodes, links, num_links);

            // Rendering step
            RenderGraph(nodes, num_nodes, links, num_links);
        } else {
            // Display welcome screen if no nodes
            BeginDrawing();
            ClearBackground(BLACK);
            DrawText("Semantic Embedding Visualization", screenWidth / 2 - 200, screenHeight / 2 - 100, 20, WHITE);
            DrawText("Press 'T' to enter text for visualizing word embeddings", screenWidth / 2 - 250, screenHeight / 2 - 50, 20, WHITE);
            DrawText("Example: \"The king and queen met with the man and woman\"", screenWidth / 2 - 270, screenHeight / 2, 20, GREEN);
            DrawText("Example: \"Red blue green yellow are colors\"", screenWidth / 2 - 200, screenHeight / 2 + 30, 20, GREEN);
            DrawText("Example: \"Dogs cats birds and fish are animals\"", screenWidth / 2 - 220, screenHeight / 2 + 60, 20, GREEN);
            EndDrawing();
        }
        
        // Render text input UI if active
        RenderTextInput(&input_state);
        
        // Display help text
        if (!input_state.is_editing && num_nodes > 0) {
            DrawText("Press 'T' to enter more text for visualization", 10, 10, 20, WHITE);
            DrawText("Press 'R' to reset and clear all nodes", 10, 40, 20, WHITE);
        }
        
        // Check for reset command
        if (IsKeyPressed(KEY_R)) {
            // Clear all nodes and links
            num_nodes = 0;
            num_links = 0;
            // Keep allocated memory but reset the count
        }
    }

    // --- Cleanup ---
    CloseWindow(); // Close Raylib window and OpenGL context
    free(nodes);
    free(links);

    return EXIT_SUCCESS;
} 