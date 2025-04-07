#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <raylib.h> // Include Raylib header

#define mass 10.0f      // Use float for physics constants
#define stiffness 100.0f   // Adjusted for potentially better visual results
#define DeltaT 0.016f    // Time step (e.g., for ~60 FPS)
#define rest_length 70.0f     // Rest length of springs
#define damping 0.98f    // Damping factor to stabilize simulation
#define repulsion_k 20000.0f // Repulsion strength constant
#define separation_stiffness 1000.0f // Force strength for overlapping nodes
#define node_radius 10.0f      // Visual radius of nodes

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
	char name[4]; // Increased size slightly for safety (e.g., "999\0")
} Node;

// Link structure
typedef struct Link {
	// Store indices directly for efficiency
	unsigned int start_idx;
	unsigned int end_idx;
	// Keep names if needed for other purposes, but indices are better for lookups
	// char depart[4];
	// char fin[4];
} Link;

// Checks if two nodes are linked based on indices
int AreNodesLinked(struct Link *links, unsigned int num_links, unsigned int idx1, unsigned int idx2) {
    for (unsigned int i = 0; i < num_links; i++) {
        if ((links[i].start_idx == idx1 && links[i].end_idx == idx2) ||
            (links[i].start_idx == idx2 && links[i].end_idx == idx1)) {
            return 1;
        }
    }
    return 0;
}

// Initializes graph nodes and reads links from the file
void InitGraph(struct Node *nodes, unsigned int num_nodes, struct Link *links, unsigned int *num_links_read, FILE *fp) {
    char s[10];
    unsigned int start_idx;
    unsigned int end_idx;
    unsigned int actual_links = 0;

    // Initialize node positions and properties
    for (unsigned int i = 0; i < num_nodes; i++) {
        nodes[i].posx = (float)((rand() % (screenWidth - 200)) + 100); // Adjust initial position based on screen size
        nodes[i].posy = (float)((rand() % (screenHeight - 200)) + 100);
        nodes[i].velx = 0.0f;
        nodes[i].vely = 0.0f;
        nodes[i].accx = 0.0f;
        nodes[i].accy = 0.0f;
        nodes[i].forcex = 0.0f;
        nodes[i].forcey = 0.0f;
        snprintf(nodes[i].name, sizeof(nodes[i].name), "%d", i);
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

// Main physics simulation step
void UpdateSimulation(struct Node *nodes, unsigned int num_nodes, struct Link *links, unsigned int num_links) {
    float dist, dist_sq, inv_dist;
    float deltaX, deltaY;
    float Fspring, Frepulsion;
    float forceX, forceY;

    // --- Reset total forces for all nodes ---
    for (unsigned int i = 0; i < num_nodes; i++) {
        nodes[i].forcex = 0.0f;
        nodes[i].forcey = 0.0f;
    }

    // --- Calculate Spring Forces (Hooke's Law for linked nodes) ---
    for (unsigned int i = 0; i < num_links; i++) {
        unsigned int idx1 = links[i].start_idx;
        unsigned int idx2 = links[i].end_idx;

        deltaX = nodes[idx1].posx - nodes[idx2].posx;
        deltaY = nodes[idx1].posy - nodes[idx2].posy;
        dist_sq = deltaX * deltaX + deltaY * deltaY;

        // Avoid division by zero and instability at very small distances
        if (dist_sq < 1e-4f) {
             dist = 0.01f; // Minimum distance
             dist_sq = dist * dist;
             // Apply a small random nudge if nodes are exactly overlapping
             deltaX = (rand() % 2 == 0 ? 1.0f : -1.0f) * dist;
             deltaY = (rand() % 2 == 0 ? 1.0f : -1.0f) * dist;
        } else {
             dist = sqrtf(dist_sq);
        }

        // Spring force: F = -k * (dist - rest_length) * (delta / dist)
        Fspring = stiffness * (dist - rest_length); // Magnitude (negative if compressed, positive if stretched)
        inv_dist = 1.0f / dist;
        forceX = Fspring * deltaX * inv_dist;
        forceY = Fspring * deltaY * inv_dist;

        // Accumulate spring forces
        nodes[idx1].forcex -= forceX; // Force on idx1 is opposite to delta vector
        nodes[idx1].forcey -= forceY;
        nodes[idx2].forcex += forceX; // Equal and opposite force on idx2
        nodes[idx2].forcey += forceY;
    }

    // --- Calculate Repulsion Forces (Coulomb-like for non-linked nodes) ---
    for (unsigned int i = 0; i < num_nodes; i++) {
        for (unsigned int j = i + 1; j < num_nodes; j++) {
            // Only apply repulsion if nodes are NOT linked
            if (!AreNodesLinked(links, num_links, i, j)) {
                deltaX = nodes[i].posx - nodes[j].posx;
                deltaY = nodes[i].posy - nodes[j].posy;
                dist_sq = deltaX * deltaX + deltaY * deltaY;

                // Clamp minimum distance squared slightly to avoid division by zero if perfectly overlapping
                if (dist_sq < 1e-6f) {
                    dist_sq = 1e-6f;
                }
                dist = sqrtf(dist_sq);

                float min_separation = 2.0f * node_radius;

                if (dist < min_separation) {
                    // Nodes are overlapping - apply strong separation force
                    float overlap = min_separation - dist;
                    float Fsep_magnitude = separation_stiffness * overlap;

                    // Avoid division by zero for direction, normalize delta vector
                    // Need a different inv_dist here to avoid potential modification below
                    float inv_dist_sep = 1.0f / dist;
                    float normX = deltaX * inv_dist_sep;
                    float normY = deltaY * inv_dist_sep;

                    forceX = Fsep_magnitude * normX;
                    forceY = Fsep_magnitude * normY;
                } else {
                    // Nodes are not overlapping - apply standard 1/dist^2 repulsion
                    // Repulsion force: F = k_rep / dist^2 * (delta / dist) = k_rep * delta / dist^3
                    // Need a different inv_dist here to avoid potential modification above
                    float inv_dist_rep = 1.0f / dist;
                    Frepulsion = repulsion_k / dist_sq; // Magnitude (always positive for repulsion)

                    forceX = Frepulsion * deltaX * inv_dist_rep;
                    forceY = Frepulsion * deltaY * inv_dist_rep;
                }

                // Accumulate forces (either separation or repulsion)
                nodes[i].forcex += forceX; // Force on i pushes away from j
                nodes[i].forcey += forceY;
                nodes[j].forcex -= forceX; // Equal and opposite force on j
                nodes[j].forcey -= forceY;
            }
        }
    }

    // --- Update physics (Velocity Verlet or Euler integration) ---
    // Using simple Euler integration
    for (unsigned int i = 0; i < num_nodes; i++) {
        // Calculate acceleration: a = F / m
        nodes[i].accx = nodes[i].forcex / mass;
        nodes[i].accy = nodes[i].forcey / mass;

        // Update velocity: v_new = (v_old + a * dt) * damping
        nodes[i].velx = (nodes[i].velx + nodes[i].accx * DeltaT) * damping;
        nodes[i].vely = (nodes[i].vely + nodes[i].accy * DeltaT) * damping;

        // Update position: x_new = x_old + v_new * dt
        nodes[i].posx += nodes[i].velx * DeltaT;
        nodes[i].posy += nodes[i].vely * DeltaT;

        // Boundary checks using Raylib screen dimensions
        if (nodes[i].posx < node_radius) { nodes[i].posx = node_radius; nodes[i].velx *= -0.5f; }
        if (nodes[i].posx > screenWidth - node_radius) { nodes[i].posx = screenWidth - node_radius; nodes[i].velx *= -0.5f; }
        if (nodes[i].posy < node_radius) { nodes[i].posy = node_radius; nodes[i].vely *= -0.5f; }
        if (nodes[i].posy > screenHeight - node_radius) { nodes[i].posy = screenHeight - node_radius; nodes[i].vely *= -0.5f; }
    }
}

// Renders the graph using Raylib functions
void RenderGraph(struct Node *graph_nodes, unsigned int num_nodes, struct Link *links, unsigned int num_links) {
    // Define colors using Raylib COLOR types
    Color nodeColor = WHITE;
    Color nodeOutlineColor = GRAY;
    Color lineColor = BLUE;
    Color textColor = BLACK;
    int fontSize = 10; // Font size for node labels

    BeginDrawing();
    ClearBackground(BLACK); // Clear with black background

    // Draw lines first
    for (unsigned int i = 0; i < num_links; i++) {
        unsigned int idx1 = links[i].start_idx;
        unsigned int idx2 = links[i].end_idx;
        // Create Vector2 for line endpoints
        Vector2 startPos = { graph_nodes[idx1].posx, graph_nodes[idx1].posy };
        Vector2 endPos = { graph_nodes[idx2].posx, graph_nodes[idx2].posy };
        DrawLineV(startPos, endPos, lineColor);
    }

    // Draw nodes and text
    for (unsigned int i = 0; i < num_nodes; i++) {
        // Create Vector2 for circle center
        Vector2 center = { graph_nodes[i].posx, graph_nodes[i].posy };

        // Draw circle outline first then filled circle
        DrawCircleV(center, node_radius + 1, nodeOutlineColor); // Slightly larger for outline
        DrawCircleV(center, node_radius, nodeColor);

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

int main() {
    FILE *fp = NULL;
    unsigned int num_nodes = 0;
    unsigned int num_links_alloc = 0;
    unsigned int num_links_actual = 0;
    char s[20];

    // --- File Reading and Graph Initialization ---
    fp = fopen("file.txt", "r");
    if (fp == NULL) {
        perror("Error opening file.txt");
        return EXIT_FAILURE;
    }

    // Read number of nodes
    if (fgets(s, sizeof(s), fp) == NULL || sscanf(s, "%u", &num_nodes) != 1 || num_nodes == 0 || num_nodes > 10000) { // Added sanity check
        fprintf(stderr, "Error reading a valid number of nodes (1-10000) from file.txt\n");
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
        fprintf(stderr,"Warning: No link lines found in file.txt\n");
     }

    rewind(fp);             // Go back to the beginning
    fgets(s, sizeof(s), fp); // Skip the first line (number of nodes) again

    // Allocate memory
    struct Node *nodes = malloc(sizeof(struct Node) * num_nodes);
    if (nodes == NULL) {
        perror("Failed to allocate memory for nodes");
        fclose(fp);
        return EXIT_FAILURE;
    }
    // Allocate based on counted lines, even if some might be invalid
    struct Link *links = NULL;
     if (num_links_alloc > 0) {
        links = malloc(sizeof(struct Link) * num_links_alloc);
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
         fprintf(stderr, "Info: Allocated for %d links, but read %d valid links.\n", num_links_alloc, num_links_actual);
         // Optional: realloc 'links' to 'num_links_actual' if memory is critical
     }


    // --- Raylib Setup ---
    InitWindow(screenWidth, screenHeight, "Bouncing Graph - Raylib");
    SetTargetFPS(60); // Set desired frame rate

    // --- Main Loop ---
    // Use Raylib's WindowShouldClose() function
    while (!WindowShouldClose()) {
        // Simulation step
        UpdateSimulation(nodes, num_nodes, links, num_links_actual);

        // Rendering step
        RenderGraph(nodes, num_nodes, links, num_links_actual);
    }

    // --- Cleanup ---
    CloseWindow(); // Close Raylib window and OpenGL context
    free(nodes);
    if (links) free(links); // Free links only if allocated

    return EXIT_SUCCESS;
} 