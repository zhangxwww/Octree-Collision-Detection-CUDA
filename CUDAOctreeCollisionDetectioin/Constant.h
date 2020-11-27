#pragma once
const int WIDTH = 800;
const int HEIGHT = 600;

const int X_SEGMENTS = 50;
const int Y_SEGMENTS = 50;

const float PI = 3.14159265358979323846f;

const float GRAVITY = 8.0f;
const float BOX_SIZE = 20.0f;

const float TIME_BETWEEN_UPDATES = 0.01f;
const int TIMER_MS = 25;

const int MAX_OCTREE_DEPTH = 6;
const int MIN_BALLS_PER_OCTREE = 3;
const int MAX_BALLS_PER_OCTREE = 6;

const float SCENE_MAX_X = BOX_SIZE / 2;
const float SCENE_MAX_Y = BOX_SIZE / 2;
const float SCENE_MAX_Z = BOX_SIZE / 2;

const int SCENE_MIN_X = -SCENE_MAX_X;
const int SCENE_MIN_Y = -SCENE_MAX_Y;
const int SCENE_MIN_Z = -SCENE_MAX_Z;

const float RADIUS = 0.2;
const float MAX_VELOCITY = 10;

