#ifndef COORD_H
#define COORD_H

#include "Python.h"
#include "structmember.h"

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include "math.h"
#endif

struct cartesianCoord {   // cartesian coordinates in 2-D

	float x, y;
	cartesianCoord& operator+=(const cartesianCoord& p) {
		x += p.x;
		y += p.y;
		return *this;
	}
	cartesianCoord& operator-=(const cartesianCoord& p) {
		x -= p.x;
		y -= p.y;
		return *this;
	}
	cartesianCoord& operator*=(float scalar) {
		x *= scalar;
		y *= scalar;
		return *this;
	}
	cartesianCoord& operator/=(float scalar) {
		x /= scalar;
		y /= scalar;
		return *this;
	}
};

struct sphericalCoord {  // spherical coordinates on unit sphere

	float theta, phi;
	float x(void) { return sin(theta) * cos(phi); }   // x-coordinate
	float y(void) { return sin(theta) * sin(phi); }   // y-coordinate
	float z(void) { return cos(theta); }                // z-coordinate
};

#endif	// COORD_H
