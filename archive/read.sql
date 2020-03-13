/*
 * Auther: Harsha HN
 * Date: 5 March 2020 
 */

-- 1. Select user profile data 
/* split
	 SELECT user_id, isActive, isProfileCompleted, lang, myStory, describeYourself, iAmCustom, meetForCustom, iAm, meetFor, marital, children
		FROM user_details
			INNER JOIN users ON user_details.user_id=users.id
*/

-- 2. Select friend links 
/* split
   SELECT user_id,friend_id FROM friends f
*/ 

-- 3. Chat connection: id, chatfriend_id, numofchatsfrom(A,B)
/* split
 	SELECT * FROM *
 */

-- 4. Activities data
/* split
 	SELECT id, title, description FROM activities
 */

-- 5. Activity links
/* 
 	SELECT activity_id, user_id FROM activity_invites
	 	WHERE isGoing=1 
 */



/*
	 SELECT user_id, myStory, describeYourself, iAmCustom, meetForCustom  FROM user_details ud 
	 SELECT user_id, isProfileCompleted, isActive FROM users u2 
	 SELECT iAm, meetFor, marital, children, lang FROM user_details ud 
*/