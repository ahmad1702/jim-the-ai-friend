import { Link } from "react-router-dom";

/**
 * To Separate Semantics from ChakraUI Link component.
 * Router Link will be a react-router link
 */
export const RouterLink = Link;

export const getAPIUrl = () => {
    return `${window.location.origin}/api/`
}